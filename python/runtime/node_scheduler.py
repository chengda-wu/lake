"""节点级调度：Host Req 权威 + continuous batching + overlap 主循环（C0 同步简化）。

参考:SGLang `managers/scheduler.py::event_loop_overlap`（CPU 收尾 ∥ GPU forward）；
vLLM `Scheduler.schedule` → `SchedulerOutput`。
lake：结束 → `pool_iface.on_request_finished`；DP sync 落本层（C0 单卡跳过）。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from engine.model_runner import ModelRunner, ModelRunnerOutput
from engine.pool_iface import PoolIface
from runtime.req import Req
from runtime.role import RoleConfig
from runtime.scheduler_output import (
    CachedRequestData,
    ForwardMode,
    NewRequestData,
    ReqIoSet,
    SamplingParams,
    SchedulerOutput,
)

LOG = logging.getLogger("lake.node_scheduler")


def mock_decode_tokens(prompt: List[int], max_new: int) -> List[int]:
    """可复现 mock:基于 prompt 末 token 递推固定序列（与旧 worker 一致）。"""
    seed = prompt[-1] if prompt else 0
    return [((seed + i + 1) % 1000) + 1000 for i in range(max_new)]


class NodeScheduler:
    def __init__(self, pool: PoolIface, runner: ModelRunner, role: Optional[RoleConfig] = None) -> None:
        self._pool = pool
        self._runner = runner
        self._role = role or RoleConfig()
        self._reqs: Dict[str, Req] = {}
        self._waiting: List[str] = []
        self._running: List[str] = []
        self._step_id = 0
        # C0：单请求 Generate 预生成的 mock 输出队列
        self._mock_remaining: Dict[str, List[int]] = {}

    def add_request(self, req: Req) -> None:
        if req.req_id in self._reqs:
            raise ValueError(f"duplicate req_id={req.req_id}")
        self._reqs[req.req_id] = req
        self._waiting.append(req.req_id)
        max_new = req.sampling_params.max_new_tokens
        self._mock_remaining[req.req_id] = mock_decode_tokens(req.prompt_token_ids, max_new)

    def run_until_idle(self) -> None:
        """同步主循环（C0：无 overlap 队列；C1 再对齐 event_loop_overlap）。"""
        while self._waiting or self._running:
            output = self.schedule()
            if output.total_num_scheduled_tokens == 0 and output.forward_mode != ForwardMode.IDLE:
                break
            ready = self._pool.prepare_step(output, self._reqs)
            runner_out = self._runner.execute_model(output, ready)
            self._process_batch_result(output, runner_out)

    def schedule(self) -> SchedulerOutput:
        self._step_id += 1
        step = self._step_id

        # 晋升 waiting → running（C0：一次一个新请求 + 已有 running）
        if self._waiting and not self._running:
            rid = self._waiting.pop(0)
            self._running.append(rid)

        if not self._running:
            return SchedulerOutput(
                step_id=step,
                forward_mode=ForwardMode.IDLE,
                total_num_scheduled_tokens=0,
            )

        new_reqs: List[NewRequestData] = []
        cached = CachedRequestData()
        num_tokens: Dict[str, int] = {}
        read_set: List[ReqIoSet] = []
        write_set: List[ReqIoSet] = []

        # C0 策略：新请求第一步 EXTEND（写满 prompt KV）；之后 DECODE 每次 1 token
        modes: List[ForwardMode] = []
        for rid in list(self._running):
            req = self._reqs[rid]
            if req.num_computed_tokens < len(req.prompt_token_ids):
                n = len(req.prompt_token_ids) - req.num_computed_tokens
                num_tokens[rid] = n
                write_set.append(
                    ReqIoSet(req_id=rid, token_start=req.num_computed_tokens, token_end=len(req.prompt_token_ids))
                )
                if req.num_computed_tokens == 0 and req.num_output_tokens == 0:
                    new_reqs.append(
                        NewRequestData(
                            req_id=rid,
                            prompt_token_ids=list(req.prompt_token_ids),
                            sampling_params=req.sampling_params,
                            num_computed_tokens=req.num_computed_tokens,
                        )
                    )
                else:
                    cached.req_ids.append(rid)
                    cached.num_computed_tokens.append(req.num_computed_tokens)
                    cached.num_output_tokens.append(req.num_output_tokens)
                modes.append(ForwardMode.EXTEND)
            else:
                remain = self._mock_remaining.get(rid) or []
                if not remain:
                    continue
                num_tokens[rid] = 1
                # decode 读已有前缀（逻辑范围）；写 1 个新 token 槽
                end = len(req.all_token_ids)
                read_set.append(ReqIoSet(req_id=rid, token_start=0, token_end=end))
                write_set.append(ReqIoSet(req_id=rid, token_start=end, token_end=end + 1))
                cached.req_ids.append(rid)
                cached.num_computed_tokens.append(req.num_computed_tokens)
                cached.num_output_tokens.append(req.num_output_tokens)
                modes.append(ForwardMode.DECODE)

        if not num_tokens:
            return SchedulerOutput(step_id=step, forward_mode=ForwardMode.IDLE, total_num_scheduled_tokens=0)

        if all(m == ForwardMode.EXTEND for m in modes):
            mode = ForwardMode.EXTEND
        elif all(m == ForwardMode.DECODE for m in modes):
            mode = ForwardMode.DECODE
        else:
            mode = ForwardMode.MIXED

        return SchedulerOutput(
            step_id=step,
            forward_mode=mode,
            scheduled_new_reqs=new_reqs,
            scheduled_cached_reqs=cached,
            num_scheduled_tokens=num_tokens,
            total_num_scheduled_tokens=sum(num_tokens.values()),
            read_set=read_set,
            write_set=write_set,
            global_num_tokens=None,
            can_run_graph=None,
        )

    def _process_batch_result(self, output: SchedulerOutput, runner_out: ModelRunnerOutput) -> None:
        for rid, n in output.num_scheduled_tokens.items():
            req = self._reqs[rid]
            if output.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED) and req.num_computed_tokens < len(
                req.prompt_token_ids
            ):
                # extend：pool_iface 已保证 KV；推进 computed
                req.num_computed_tokens = len(req.prompt_token_ids)
                if output.forward_mode == ForwardMode.EXTEND:
                    continue

            # decode：消费 mock 队列（保证与旧 P3 输出一致）
            if rid in runner_out.next_token_ids or output.forward_mode in (
                ForwardMode.DECODE,
                ForwardMode.MIXED,
            ):
                remain = self._mock_remaining.get(rid) or []
                if remain:
                    tok = remain.pop(0)
                    req.output_token_ids.append(tok)
                    req.num_computed_tokens += 1
                    self._mock_remaining[rid] = remain

            if req.num_output_tokens >= req.sampling_params.max_new_tokens:
                req.finished = True
                req.finish_reason = "length"
                self._finish_req(rid)

    def _finish_req(self, rid: str) -> None:
        req = self._reqs[rid]
        if rid in self._running:
            self._running.remove(rid)
        self._pool.on_request_finished(req)
        LOG.info("finished req_id=%s reason=%s out=%d", rid, req.finish_reason, req.num_output_tokens)

    def get_req(self, req_id: str) -> Req:
        return self._reqs[req_id]


def build_req_from_generate(
    request_id: str,
    model_id: str,
    prompt_tokens: List[int],
    max_new_tokens: int,
    node_id: str,
) -> Req:
    return Req(
        req_id=request_id,
        model_id=model_id or "mock-llm",
        prompt_token_ids=list(prompt_tokens),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens or 4),
        node_id=node_id,
    )
