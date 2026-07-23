"""节点级调度：Host Req 权威 + continuous batching + overlap 主循环。

参考:SGLang `managers/scheduler.py::event_loop_overlap` + `overlap_utils.FutureMap`；
vLLM `Scheduler.schedule` → `SchedulerOutput`。
lake：结束 → `pool_iface.on_request_finished`；DP sync 落本层（单卡跳过）。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from engine.model_runner import ModelRunner, ModelRunnerOutput
from engine.pool_iface import PoolIface
from engine.pool_types import ReadyHandle
from runtime.exec_mode import ExecMode
from runtime.future_map import FutureMap
from runtime.mode_select import full_local_hit, select_exec_mode
from runtime.prefix_hint import PrefixHint
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


@dataclass
class _BatchResult:
    output: SchedulerOutput
    runner_out: ModelRunnerOutput
    ready: ReadyHandle


class NodeScheduler:
    def __init__(self, pool: PoolIface, runner: ModelRunner, role: Optional[RoleConfig] = None) -> None:
        self._pool = pool
        self._runner = runner
        self._role = role or RoleConfig()
        self._reqs: Dict[str, Req] = {}
        self._waiting: List[str] = []
        self._running: List[str] = []
        self._step_id = 0
        self._mock_remaining: Dict[str, List[int]] = {}
        # 已 schedule、尚未 process 的 decode token 数（overlap 下 Host Req 滞后一步）
        self._inflight_decode: Dict[str, int] = {}
        self._future_map = FutureMap()
        self._result_queue: Deque[_BatchResult] = deque()
        # C4：上步 pre_forward 产出、待 TARGET_VERIFY 的 draft
        self._pending_drafts: Dict[str, List[int]] = {}
        # 测试钩子：记录 execute / process 时序
        self.timeline: List[Tuple[str, int]] = []

    @property
    def _use_runner_tokens(self) -> bool:
        return self._role.model_backend == "tiny_lm"

    @property
    def _spec_enabled(self) -> bool:
        return self._role.enable_drafter and self._use_runner_tokens

    @property
    def future_map(self) -> FutureMap:
        return self._future_map

    def add_request(self, req: Req, hint: Optional[PrefixHint] = None) -> None:
        if req.req_id in self._reqs:
            raise ValueError(f"duplicate req_id={req.req_id}")
        if hint is not None:
            req.apply_prefix_hint(hint)
            req.exec_mode = select_exec_mode(
                hint, prompt_len=len(req.prompt_token_ids), role=self._role.role
            )
            if full_local_hit(hint, len(req.prompt_token_ids)):
                # vLLM 几何：整段已在 L0 → computed=prompt_len，下一步直接生成
                req.num_computed_tokens = len(req.prompt_token_ids)
        self._reqs[req.req_id] = req
        self._waiting.append(req.req_id)
        max_new = req.sampling_params.max_new_tokens
        if self._role.model_backend == "mock":
            self._mock_remaining[req.req_id] = mock_decode_tokens(req.prompt_token_ids, max_new)
        else:
            self._mock_remaining[req.req_id] = []

    def run_until_idle(self) -> None:
        """主循环：默认 overlap（对齐 SGLang event_loop_overlap）。"""
        if self._role.enable_overlap:
            self._event_loop_overlap()
        else:
            self._event_loop_normal()

    def _event_loop_normal(self) -> None:
        while self._waiting or self._running or self._result_queue:
            output = self.schedule()
            if output.total_num_scheduled_tokens == 0:
                self._drain_results()
                if not self._waiting and not self._running:
                    break
                continue
            self._run_batch(output)
            self._pop_and_process()

    def _event_loop_overlap(self) -> None:
        """
        while True:
          schedule
          if disable_overlap: drain
          run_batch → result_queue
          process 上批（与本批 forward 重叠）
        """
        while True:
            disable = self._should_disable_overlap()
            if disable:
                self._drain_results()

            output = self.schedule()
            if output.total_num_scheduled_tokens == 0:
                self._drain_results()
                if not self._waiting and not self._running and not self._result_queue:
                    break
                continue

            self._run_batch(output)

            if not disable and len(self._result_queue) > 1:
                self._pop_and_process()

        self._drain_results()

    def _should_disable_overlap(self) -> bool:
        """关 overlap 例外（C1：连续 EXTEND 可选；spec+grammar 留给 D7）。"""
        # 若 waiting 里将有 extend、且队列非空，可强制同步以利 TTFT（对齐
        # SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP 精神）。C1 默认不强制。
        return False

    def _run_batch(self, output: SchedulerOutput) -> None:
        ready = self._pool.prepare_step(output, self._reqs)
        # D2：allow_partial_hit 缩批后须按 effective_* 执行（默认与 plan 相同）
        output = self._respect_effective_sets(output, ready)
        self.timeline.append(("execute", output.step_id))
        runner_out = self._runner.execute_model(output, ready, self._reqs)
        self._result_queue.append(_BatchResult(output=output, runner_out=runner_out, ready=ready))

    def _respect_effective_sets(self, output: SchedulerOutput, ready: ReadyHandle) -> SchedulerOutput:
        """按 ReadyHandle.effective_*_set 过滤本步；丢弃的 req 回滚 inflight，下步重试。

        effective_* = None ⇒ agent 未填（FakePool/旧 agent）→ 未缩批，原样执行。
        effective_* = []  ⇒ agent 显式缩批至空（allow_partial_hit 把全批丢掉）→ 降为 IDLE。
        """
        if ready.effective_read_set is None and ready.effective_write_set is None:
            return output
        eff_r = ready.effective_read_set if ready.effective_read_set is not None else output.read_set
        eff_w = ready.effective_write_set if ready.effective_write_set is not None else output.write_set
        keep = {io.req_id for io in eff_r} | {io.req_id for io in eff_w}
        planned = set(output.num_scheduled_tokens)
        dropped = planned - keep
        if not dropped:
            return output

        for rid in dropped:
            n = output.num_scheduled_tokens.get(rid, 0)
            if n:
                self._inflight_decode[rid] = max(0, self._inflight_decode.get(rid, 0) - n)
            LOG.info("respect effective_* drop req=%s step=%s", rid, output.step_id)

        num_tokens = {rid: n for rid, n in output.num_scheduled_tokens.items() if rid in keep}
        if not num_tokens:
            return SchedulerOutput(
                step_id=output.step_id,
                forward_mode=ForwardMode.IDLE,
                total_num_scheduled_tokens=0,
            )

        new_reqs = [r for r in output.scheduled_new_reqs if r.req_id in keep]
        cached = CachedRequestData()
        old_c = output.scheduled_cached_reqs
        for i, rid in enumerate(old_c.req_ids):
            if rid not in keep:
                continue
            cached.req_ids.append(rid)
            if i < len(old_c.num_computed_tokens):
                cached.num_computed_tokens.append(old_c.num_computed_tokens[i])
            if i < len(old_c.num_output_tokens):
                cached.num_output_tokens.append(old_c.num_output_tokens[i])

        req_modes = {rid: m for rid, m in output.req_forward_modes.items() if rid in keep}
        modes = list(req_modes.values())
        if all(m == ForwardMode.EXTEND for m in modes):
            mode = ForwardMode.EXTEND
        elif all(m == ForwardMode.DECODE for m in modes):
            mode = ForwardMode.DECODE
        elif all(m == ForwardMode.TARGET_VERIFY for m in modes):
            mode = ForwardMode.TARGET_VERIFY
        else:
            mode = ForwardMode.MIXED

        spec = None
        if output.scheduled_spec_decode_tokens:
            spec = {rid: t for rid, t in output.scheduled_spec_decode_tokens.items() if rid in keep}
        computed_at = {
            rid: c for rid, c in output.req_num_computed_at_schedule.items() if rid in keep
        }

        return SchedulerOutput(
            step_id=output.step_id,
            forward_mode=mode,
            scheduled_new_reqs=new_reqs,
            scheduled_cached_reqs=cached,
            num_scheduled_tokens=num_tokens,
            total_num_scheduled_tokens=sum(num_tokens.values()),
            read_set=[io for io in eff_r if io.req_id in keep],
            write_set=[io for io in eff_w if io.req_id in keep],
            global_num_tokens=output.global_num_tokens,
            can_run_graph=output.can_run_graph,
            req_forward_modes=req_modes,
            scheduled_spec_decode_tokens=spec or None,
            has_structured_output=output.has_structured_output,
            req_num_computed_at_schedule=computed_at,
        )

    def _pop_and_process(self) -> None:
        if not self._result_queue:
            return
        batch = self._result_queue.popleft()
        self.timeline.append(("process", batch.output.step_id))
        self._apply_ready_stats(batch.ready)
        self._process_batch_result(batch.output, batch.runner_out)
        self._future_map.publish()

    def _apply_ready_stats(self, ready: ReadyHandle) -> None:
        """agent 只回 StepStats；Host Req 复用字段在此统一写入。

        仅采纳带 prefill_blocks 的步（EXTEND/前缀 ensure）。decode 步常带回
        reused>0 的回声或空统计，不得覆盖冷启动的 reused==0。
        """
        for rid, st in ready.stats_by_req.items():
            req = self._reqs.get(rid)
            if req is None:
                continue
            if st.prefill_blocks:
                req.prefill_blocks = st.prefill_blocks
                req.reused_blocks = st.reused_blocks

    def _drain_results(self) -> None:
        while self._result_queue:
            self._pop_and_process()

    def schedule(self) -> SchedulerOutput:
        self._step_id += 1
        step = self._step_id

        # continuous batching：填满 running 槽
        while self._waiting and len(self._running) < self._role.max_running_reqs:
            self._running.append(self._waiting.pop(0))

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
        req_modes: Dict[str, ForwardMode] = {}
        computed_at: Dict[str, int] = {}
        spec_tokens: Dict[str, List[int]] = {}

        for rid in list(self._running):
            req = self._reqs[rid]
            if req.finished:
                continue
            prompt_len = len(req.prompt_token_ids)
            computed = req.num_computed_tokens
            computed_at[rid] = computed

            if computed < prompt_len:
                # prompt 残差不可重叠重入（Host computed 未推进会双写）
                if self._has_unprocessed_prompt(rid, prompt_len):
                    continue
                n = prompt_len - computed
                num_tokens[rid] = n
                if computed > 0:
                    read_set.append(ReqIoSet(req_id=rid, token_start=0, token_end=computed))
                write_set.append(ReqIoSet(req_id=rid, token_start=computed, token_end=prompt_len))
                if computed == 0 and req.num_output_tokens == 0:
                    new_reqs.append(
                        NewRequestData(
                            req_id=rid,
                            prompt_token_ids=list(req.prompt_token_ids),
                            sampling_params=req.sampling_params,
                            num_computed_tokens=computed,
                        )
                    )
                else:
                    cached.req_ids.append(rid)
                    cached.num_computed_tokens.append(computed)
                    cached.num_output_tokens.append(req.num_output_tokens)
                req_modes[rid] = ForwardMode.EXTEND  # 派生标签：prompt 相
                continue

            # 生成相：允许 overlap；用 _inflight_decode 预留，不因 result_queue 挡 schedule
            inflight = self._inflight_decode.get(rid, 0)
            if self._use_runner_tokens:
                left = req.sampling_params.max_new_tokens - req.num_output_tokens - inflight
                if left <= 0:
                    continue
            else:
                remain = self._mock_remaining.get(rid) or []
                if len(remain) <= inflight:
                    continue

            pending = self._pending_drafts.get(rid) or []
            _ = self._future_map.resolve(rid)
            end = len(req.all_token_ids) + inflight
            read_set.append(ReqIoSet(req_id=rid, token_start=0, token_end=end))
            cached.req_ids.append(rid)
            cached.num_computed_tokens.append(computed + inflight)
            cached.num_output_tokens.append(req.num_output_tokens + inflight)

            if self._spec_enabled and pending:
                max_accept = min(
                    len(pending) + 1, left if self._use_runner_tokens else len(pending) + 1
                )
                num_tokens[rid] = max_accept
                write_set.append(ReqIoSet(req_id=rid, token_start=end, token_end=end + max_accept))
                req_modes[rid] = ForwardMode.TARGET_VERIFY
                spec_tokens[rid] = list(pending)
                self._inflight_decode[rid] = inflight + max_accept
            else:
                num_tokens[rid] = 1
                write_set.append(ReqIoSet(req_id=rid, token_start=end, token_end=end + 1))
                req_modes[rid] = ForwardMode.DECODE
                self._inflight_decode[rid] = inflight + 1

        if not num_tokens:
            return SchedulerOutput(step_id=step, forward_mode=ForwardMode.IDLE, total_num_scheduled_tokens=0)

        modes = list(req_modes.values())
        if all(m == ForwardMode.EXTEND for m in modes):
            mode = ForwardMode.EXTEND
        elif all(m == ForwardMode.DECODE for m in modes):
            mode = ForwardMode.DECODE
        elif all(m == ForwardMode.TARGET_VERIFY for m in modes):
            mode = ForwardMode.TARGET_VERIFY
        else:
            mode = ForwardMode.MIXED

        # 全 1-token 生成步可图（骨架占位）
        can_graph = mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY) and all(
            n == 1 or req_modes[r] == ForwardMode.TARGET_VERIFY for r, n in num_tokens.items()
        )

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
            can_run_graph=can_graph,
            req_forward_modes=req_modes,
            scheduled_spec_decode_tokens=spec_tokens or None,
            req_num_computed_at_schedule={r: computed_at[r] for r in num_tokens},
        )

    def _has_unprocessed_prompt(self, rid: str, prompt_len: int) -> bool:
        for br in self._result_queue:
            if rid not in br.output.num_scheduled_tokens:
                continue
            c = br.output.req_num_computed_at_schedule.get(rid, 0)
            if c < prompt_len:
                return True
        return False

    def _process_batch_result(self, output: SchedulerOutput, runner_out: ModelRunnerOutput) -> None:
        for rid, scheduled_n in output.num_scheduled_tokens.items():
            req = self._reqs[rid]
            prompt_len = len(req.prompt_token_ids)
            computed_before = output.req_num_computed_at_schedule.get(rid, req.num_computed_tokens)
            spec = (output.scheduled_spec_decode_tokens or {}).get(rid)

            if computed_before < prompt_len:
                # prompt 残差步：推进 computed；不产出 user-facing token
                req.num_computed_tokens = prompt_len
                drafts = runner_out.next_draft_tokens.get(rid) or []
                if drafts:
                    self._pending_drafts[rid] = drafts
                continue

            # 生成步
            self._inflight_decode[rid] = max(0, self._inflight_decode.get(rid, 0) - scheduled_n)
            if spec is not None:
                self._pending_drafts.pop(rid, None)

            if self._use_runner_tokens:
                produced = list(runner_out.next_token_ids.get(rid) or [])
                left = req.sampling_params.max_new_tokens - req.num_output_tokens
                if left < len(produced):
                    produced = produced[:left]
                for tok in produced:
                    req.output_token_ids.append(int(tok))
                    req.num_computed_tokens += 1
                if produced:
                    self._future_map.stash(rid, int(produced[-1]))
                # D10：InMemory 绝对值 commit × overlap 不安全；见 compute-layer D10
                self._pool.commit_write_extent(rid, len(req.all_token_ids))
                next_d = runner_out.next_draft_tokens.get(rid) or []
                if next_d and not req.finished:
                    self._pending_drafts[rid] = next_d
            else:
                remain = self._mock_remaining.get(rid) or []
                if remain:
                    tok = remain.pop(0)
                    req.output_token_ids.append(tok)
                    req.num_computed_tokens += 1
                    self._mock_remaining[rid] = remain
                    self._future_map.stash(rid, tok)
                self._pool.commit_write_extent(rid, len(req.all_token_ids))

            if req.num_output_tokens >= req.sampling_params.max_new_tokens:
                req.finished = True
                req.finish_reason = "length"
                self._finish_req(rid)

    def _finish_req(self, rid: str) -> None:
        req = self._reqs[rid]
        if rid in self._running:
            self._running.remove(rid)
        self._inflight_decode.pop(rid, None)
        self._pending_drafts.pop(rid, None)
        self._future_map.clear(rid)
        self._runner.clear_drafter(rid)
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
