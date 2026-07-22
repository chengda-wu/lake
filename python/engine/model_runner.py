"""薄 ModelRunner：consume ready → forward → done →（sample 由 scheduler 编排）。

对齐 vLLM `GPUModelRunner.execute_model` 的 step 入口外形；
无 `RequestState` / 无 block_table apply（对齐 SGLang 薄 runner + lake Q1/Q2）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from engine.input_batch import InputBatch
from engine.pool_iface import PoolIface
from engine.pool_types import ReadyHandle
from runtime.scheduler_output import ForwardMode, SchedulerOutput


@dataclass
class ModelRunnerOutput:
    """本步 GPU/mock 前向结果；采样前的 logits 占位用 token 建议。"""

    step_id: int
    # req_id → 本步建议产出的 token（C0 mock：固定递推）
    next_token_ids: Dict[str, List[int]] = field(default_factory=dict)


class ModelRunner:
    def __init__(self, pool: PoolIface) -> None:
        self._pool = pool
        self._input_batch = InputBatch()

    def execute_model(self, output: SchedulerOutput, ready: ReadyHandle) -> ModelRunnerOutput:
        if ready.step_id != output.step_id:
            raise RuntimeError(f"ready/output step mismatch: {ready.step_id} vs {output.step_id}")

        self._input_batch.req_ids = list(output.num_scheduled_tokens.keys())

        # C0：无真模型。decode/mixed 产出 1 token；extend-only 不采样（由 scheduler 决定）。
        next_tokens: Dict[str, List[int]] = {}
        if output.forward_mode in (ForwardMode.DECODE, ForwardMode.MIXED, ForwardMode.PREBUILT):
            for req_id, n in output.num_scheduled_tokens.items():
                if n <= 0:
                    continue
                # 真实路径：读固定 buffer + block table；此处占位 1 token/步
                next_tokens[req_id] = [0]  # 占位，scheduler 用 mock 策略覆盖

        self._pool.done(output.step_id)
        return ModelRunnerOutput(step_id=output.step_id, next_token_ids=next_tokens)
