"""薄 ModelRunner：consume ready → forward → done →（sample / 投机同路径）。

对齐 vLLM `GPUModelRunner.execute_model`；投机编排仿 SGLang 共置串行
（target → post_forward → 下轮 pre_forward → target_verify）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from engine.drafter.tiny_mtp import TinyMTPDrafter
from engine.input_batch import InputBatch
from engine.models.tiny_lm import TinyLM
from engine.pool_iface import PoolIface
from engine.pool_types import ReadyHandle
from engine.sample.greedy import greedy_sample
from engine.sample.reject import chain_reject_sample
from runtime.req import Req
from runtime.scheduler_output import ForwardMode, SchedulerOutput


@dataclass
class ModelRunnerOutput:
    step_id: int
    next_token_ids: Dict[str, List[int]] = field(default_factory=dict)
    # 供下一步 TARGET_VERIFY
    next_draft_tokens: Dict[str, List[int]] = field(default_factory=dict)
    model_backend: str = "mock"


class ModelRunner:
    def __init__(
        self,
        pool: PoolIface,
        *,
        model_backend: str = "mock",
        tiny_lm: Optional[TinyLM] = None,
        enable_drafter: bool = False,
        num_draft_tokens: int = 2,
        drafter: Optional[TinyMTPDrafter] = None,
    ) -> None:
        self._pool = pool
        self._input_batch = InputBatch()
        self.model_backend = model_backend
        self._tiny: Optional[TinyLM] = tiny_lm
        if model_backend == "tiny_lm" and self._tiny is None:
            self._tiny = TinyLM()
        self.enable_drafter = enable_drafter
        self._drafter: Optional[TinyMTPDrafter] = drafter
        if enable_drafter and self._drafter is None and self._tiny is not None:
            self._drafter = TinyMTPDrafter(
                num_draft_tokens=num_draft_tokens,
                vocab_size=self._tiny.vocab_size,
                d_model=self._tiny.d_model,
                n_heads=self._tiny.n_heads,
            )

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
        next_drafts: Dict[str, List[int]] = {}

        if self.model_backend == "tiny_lm":
            next_tokens, next_drafts = self._forward_tiny(output, host_reqs or {})
        elif output.forward_mode in (
            ForwardMode.DECODE,
            ForwardMode.MIXED,
            ForwardMode.PREBUILT,
            ForwardMode.TARGET_VERIFY,
        ):
            for req_id, n in output.num_scheduled_tokens.items():
                if n <= 0:
                    continue
                phase = output.req_forward_modes.get(req_id, output.forward_mode)
                if phase in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
                    next_tokens[req_id] = [0]

        self._pool.done(output.step_id)
        return ModelRunnerOutput(
            step_id=output.step_id,
            next_token_ids=next_tokens,
            next_draft_tokens=next_drafts,
            model_backend=self.model_backend,
        )

    def _forward_tiny(
        self,
        output: SchedulerOutput,
        host_reqs: Mapping[str, Req],
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        assert self._tiny is not None
        out: Dict[str, List[int]] = {}
        drafts_out: Dict[str, List[int]] = {}
        spec_map = output.scheduled_spec_decode_tokens or {}

        for req_id, n in output.num_scheduled_tokens.items():
            if n <= 0:
                continue
            phase = output.req_forward_modes.get(req_id, output.forward_mode)
            req = host_reqs.get(req_id)
            if req is None:
                continue

            if phase == ForwardMode.EXTEND:
                # 残差：只对未计算尾做前向（整段重算简化：仍喂全 prompt）
                _ = self._tiny.forward_logits(req.prompt_token_ids)
                if self._drafter is not None:
                    self._drafter.post_forward(req_id, req.prompt_token_ids)
                    drafts_out[req_id] = self._drafter.pre_forward(req_id)
                continue

            if phase == ForwardMode.PREBUILT:
                # KV 已在 L0：跳过 prefill forward（对齐 SGLang PrebuiltExtendBatch）
                if self._drafter is not None:
                    self._drafter.post_forward(req_id, req.prompt_token_ids)
                    drafts_out[req_id] = self._drafter.pre_forward(req_id)
                continue

            if phase == ForwardMode.DRAFT_EXTEND:
                # 前缀命中后重建 draft seed（对齐 SGLang draft-extend 语义骨架）
                if self._drafter is not None:
                    self._drafter.post_forward(req_id, req.all_token_ids)
                    drafts_out[req_id] = self._drafter.pre_forward(req_id)
                continue

            if phase == ForwardMode.TARGET_VERIFY:
                draft = list(spec_map.get(req_id) or [])
                accepted = chain_reject_sample(
                    req.all_token_ids, draft, self._tiny.greedy_token
                )
                out[req_id] = accepted
            elif phase == ForwardMode.DECODE:
                logits = self._tiny.forward_logits(req.all_token_ids)
                out[req_id] = [greedy_sample(logits)]
            else:
                continue

            if self._drafter is not None and req_id in out:
                new_ctx = list(req.all_token_ids) + list(out[req_id])
                self._drafter.post_forward(req_id, new_ctx)
                drafts_out[req_id] = self._drafter.pre_forward(req_id)

        return out, drafts_out

    def clear_drafter(self, req_id: str) -> None:
        if self._drafter is not None:
            self._drafter.clear(req_id)
