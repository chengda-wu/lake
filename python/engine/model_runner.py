"""薄 ModelRunner：consume ready → forward → done →（sample 可同路径）。

对齐 vLLM `GPUModelRunner.execute_model` 的 step 入口外形；
无 `RequestState` / 无 block_table apply（对齐 SGLang 薄 runner + lake Q1/Q2）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from engine.input_batch import InputBatch
from engine.models.tiny_lm import TinyLM
from engine.pool_iface import PoolIface
from engine.pool_types import ReadyHandle
from engine.sample.greedy import greedy_sample
from runtime.req import Req
from runtime.scheduler_output import ForwardMode, SchedulerOutput


@dataclass
class ModelRunnerOutput:
    """本步前向结果。"""

    step_id: int
    next_token_ids: Dict[str, List[int]] = field(default_factory=dict)
    # 调试：backend 名
    model_backend: str = "mock"


class ModelRunner:
    def __init__(
        self,
        pool: PoolIface,
        *,
        model_backend: str = "mock",
        tiny_lm: Optional[TinyLM] = None,
    ) -> None:
        self._pool = pool
        self._input_batch = InputBatch()
        self.model_backend = model_backend
        self._tiny: Optional[TinyLM] = tiny_lm
        if model_backend == "tiny_lm" and self._tiny is None:
            self._tiny = TinyLM()

    def execute_model(
        self,
        output: SchedulerOutput,
        ready: ReadyHandle,
        host_reqs: Optional[Mapping[str, Req]] = None,
    ) -> ModelRunnerOutput:
        if ready.step_id != output.step_id:
            raise RuntimeError(f"ready/output step mismatch: {ready.step_id} vs {output.step_id}")

        self._input_batch.req_ids = list(output.num_scheduled_tokens.keys())
        next_tokens: Dict[str, List[int]] = {}

        if self.model_backend == "tiny_lm":
            next_tokens = self._forward_tiny(output, host_reqs or {})
        elif output.forward_mode in (ForwardMode.DECODE, ForwardMode.MIXED, ForwardMode.PREBUILT):
            for req_id, n in output.num_scheduled_tokens.items():
                if n <= 0:
                    continue
                phase = output.req_forward_modes.get(req_id, output.forward_mode)
                if phase == ForwardMode.DECODE:
                    next_tokens[req_id] = [0]  # mock 占位；scheduler 用 mock_remaining 覆盖

        self._pool.done(output.step_id)
        return ModelRunnerOutput(
            step_id=output.step_id,
            next_token_ids=next_tokens,
            model_backend=self.model_backend,
        )

    def _forward_tiny(
        self,
        output: SchedulerOutput,
        host_reqs: Mapping[str, Req],
    ) -> Dict[str, List[int]]:
        assert self._tiny is not None
        out: Dict[str, List[int]] = {}
        for req_id, n in output.num_scheduled_tokens.items():
            if n <= 0:
                continue
            phase = output.req_forward_modes.get(req_id, output.forward_mode)
            req = host_reqs.get(req_id)
            if req is None:
                continue
            if phase == ForwardMode.EXTEND:
                # 预热前向（验证链路）；不产出 token
                _ = self._tiny.forward_logits(req.prompt_token_ids)
                continue
            if phase == ForwardMode.DECODE:
                # 整段重算（C3）；生产改为读池 arena + block table
                ctx = req.all_token_ids
                logits = self._tiny.forward_logits(ctx)
                tok = greedy_sample(logits)
                out[req_id] = [tok]
        return out
