"""薄 ModelRunner：consume ready → prepare → forward → done → sample。

对齐 vLLM `GPUModelRunner.execute_model`：统一入口，按本步 token 几何执行
（`num_scheduled_tokens` / `req_num_computed_at_schedule`），不按 SGLang 分相状态机。
C8：`prepare_inputs` / `prepare_attn` / `sample_tokens` 拆步；残差 query 路径。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from engine.attn.metadata import AttentionMetadata, build_attn_metadata
from engine.drafter.tiny_mtp import TinyMTPDrafter
from engine.input_batch import InputBatch, InputBuffers
from engine.models.tiny_lm import TinyLM
from engine.pool_iface import PoolIface
from engine.pool_types import ReadyHandle
from engine.sample.greedy import greedy_sample
from engine.sample.grammar import apply_token_bitmask
from engine.sample.reject import chain_reject_sample
from runtime.req import Req
from runtime.scheduler_output import ForwardMode, SamplingParams, SchedulerOutput


@dataclass
class ModelRunnerOutput:
    step_id: int
    next_token_ids: Dict[str, List[int]] = field(default_factory=dict)
    next_draft_tokens: Dict[str, List[int]] = field(default_factory=dict)
    model_backend: str = "mock"


@dataclass(frozen=True)
class ModelLoadInfo:
    model_id: str
    revision: str
    backend: str
    load_dummy_weights: bool = False
    weight_pinned: bool = False


@dataclass(frozen=True)
class ModelRunnerStatus:
    model_id: str
    revision: str
    backend: str
    loaded: bool
    warmed: bool


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
        weight_pin_callback: Optional[Callable[[ModelLoadInfo], None]] = None,
    ) -> None:
        self._pool = pool
        self._input_batch = InputBatch()
        self._input_buffers = InputBuffers(max_num_reqs=64, max_num_tokens=8192)
        self._attn_meta: Optional[AttentionMetadata] = None
        self.model_backend = model_backend
        self._tiny: Optional[TinyLM] = tiny_lm
        self.enable_drafter = enable_drafter
        self._num_draft_tokens = num_draft_tokens
        self._drafter: Optional[TinyMTPDrafter] = drafter
        self._weight_pin_callback = weight_pin_callback
        self._model_id = ""
        self._model_revision = ""
        self._model_loaded = False
        self._model_warmed = False
        self._ensure_backend_initialized()

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def model_warmed(self) -> bool:
        return self._model_warmed

    @property
    def model_id(self) -> str:
        return self._model_id

    def status(self) -> ModelRunnerStatus:
        return ModelRunnerStatus(
            model_id=self._model_id,
            revision=self._model_revision,
            backend=self.model_backend,
            loaded=self._model_loaded,
            warmed=self._model_warmed,
        )

    def _ensure_backend_initialized(self) -> None:
        if self.model_backend == "tiny_lm" and self._tiny is None:
            self._tiny = TinyLM()
        if self.enable_drafter and self._drafter is None and self._tiny is not None:
            self._drafter = TinyMTPDrafter(
                num_draft_tokens=self._num_draft_tokens,
                vocab_size=self._tiny.vocab_size,
                d_model=self._tiny.d_model,
                n_heads=self._tiny.n_heads,
            )

    def load_model(
        self,
        *,
        model_id: str = "mock-llm",
        revision: str = "",
        load_dummy_weights: bool = False,
        pin_weights: bool = True,
    ) -> ModelLoadInfo:
        """C12：真实模型加载骨架。

        对齐 vLLM `GPUModelRunner.load_model` 的阶段边界：先建立模型对象，
        再初始化依赖模型的执行组件。lake 只记录状态并触发权重 pin 回调；
        权重所有权仍归存储池。
        """

        self._ensure_backend_initialized()
        self._model_id = model_id or "mock-llm"
        self._model_revision = revision
        self._model_loaded = True
        self._model_warmed = False
        info = ModelLoadInfo(
            model_id=self._model_id,
            revision=self._model_revision,
            backend=self.model_backend,
            load_dummy_weights=load_dummy_weights,
            weight_pinned=pin_weights,
        )
        if pin_weights and self._weight_pin_callback is not None:
            self._weight_pin_callback(info)
        return info

    def warmup(
        self,
        *,
        num_reqs: int = 1,
        tokens_per_req: int = 1,
    ) -> ModelRunnerOutput:
        """C12：warmup 复用生产 dummy 入口，但跳过 pool.done。"""

        if not self._model_loaded:
            self.load_model(model_id=self._model_id or "mock-llm")
        out = self.dummy_run(
            num_reqs=num_reqs,
            tokens_per_req=tokens_per_req,
            step_id=-1,
        )
        self._model_warmed = True
        return out

    def prepare_inputs(
        self,
        output: SchedulerOutput,
        host_reqs: Mapping[str, Req],
    ) -> InputBatch:
        """对齐 vLLM `prepare_inputs`：组本步 InputBatch（无跨步 RequestState）。"""
        batch = InputBatch()
        spec_map = output.scheduled_spec_decode_tokens or {}
        for req_id, n in output.num_scheduled_tokens.items():
            if n <= 0:
                continue
            req = host_reqs.get(req_id)
            if req is None:
                continue
            prompt_len = len(req.prompt_token_ids)
            computed = output.req_num_computed_at_schedule.get(req_id, req.num_computed_tokens)
            batch.req_ids.append(req_id)
            batch.num_scheduled_tokens[req_id] = n
            batch.num_computed_tokens[req_id] = computed
            if computed < prompt_len:
                end = min(prompt_len, computed + n)
                batch.token_ids[req_id] = list(req.prompt_token_ids[:end])
                batch.query_start[req_id] = computed
                batch.query_end[req_id] = end
                batch.is_prompt_phase[req_id] = True
            else:
                ctx = list(req.all_token_ids)
                draft = list(spec_map.get(req_id) or [])[: max(0, n - 1)]
                if draft:
                    # TARGET_VERIFY：输入 last_ctx + draft，产生 draft 校验 + bonus。
                    batch.token_ids[req_id] = ctx + draft
                    batch.query_start[req_id] = max(0, len(ctx) - 1)
                    batch.query_end[req_id] = len(ctx) + len(draft)
                else:
                    batch.token_ids[req_id] = ctx
                    # decode：以最后已有 token 为 query，预测下一 token
                    qs = max(0, len(ctx) - 1)
                    batch.query_start[req_id] = qs
                    batch.query_end[req_id] = len(ctx)
                batch.is_prompt_phase[req_id] = False
        self._input_batch = batch
        return batch

    def prepare_attn(self, batch: InputBatch, ready: ReadyHandle) -> AttentionMetadata:
        """对齐 vLLM `prepare_attn`：几何 + agent block table（只读）。"""
        seq_lens = {}
        for rid in batch.req_ids:
            seq_lens[rid] = batch.query_end[rid]
        self._input_buffers.materialize(
            batch, slot_mapping_by_req=ready.slot_mapping_by_req
        )
        meta = build_attn_metadata(
            seq_lens=seq_lens,
            query_start=batch.query_start,
            query_end=batch.query_end,
            block_tables=ready.block_table_by_req,
            buffers=self._input_buffers,
            req_order=batch.req_ids,
        )
        self._attn_meta = meta
        return meta

    def sample_tokens(
        self,
        output: SchedulerOutput,
        host_reqs: Mapping[str, Req],
        last_logits: Dict[str, List[float]],
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """从本步 logits 采样；接口预留与 `execute_model` 拆分（对齐 V2）。"""
        out: Dict[str, List[int]] = {}
        drafts_out: Dict[str, List[int]] = {}
        spec_map = output.scheduled_spec_decode_tokens or {}
        grammar = output.grammar_output
        deferred = set(grammar.deferred_req_ids if grammar is not None else [])
        bitmasks = grammar.token_bitmask_by_req if grammar is not None else {}
        for req_id, logits in last_logits.items():
            if req_id in deferred:
                continue
            req = host_reqs.get(req_id)
            if req is None:
                continue
            draft = list(spec_map.get(req_id) or [])
            if draft:
                assert self._tiny is not None
                accepted = chain_reject_sample(
                    req.all_token_ids, draft, self._tiny.greedy_token
                )
                out[req_id] = accepted
            else:
                masked_logits = logits
                if req_id in bitmasks:
                    masked_logits = apply_token_bitmask(logits, bitmasks[req_id])
                out[req_id] = [greedy_sample(masked_logits)]
            if self._drafter is not None and req_id in out:
                new_ctx = list(req.all_token_ids) + list(out[req_id])
                self._drafter.post_forward(req_id, new_ctx)
                drafts_out[req_id] = self._drafter.pre_forward(req_id)
        return out, drafts_out

    def execute_model(
        self,
        output: SchedulerOutput,
        ready: ReadyHandle,
        host_reqs: Optional[Mapping[str, Req]] = None,
    ) -> ModelRunnerOutput:
        if ready.step_id != output.step_id:
            raise RuntimeError(f"ready/output step mismatch: {ready.step_id} vs {output.step_id}")

        host = host_reqs or {}
        try:
            return self._execute_prepared(output, ready, host)
        finally:
            self._pool.done(output.step_id)

    def _execute_prepared(
        self,
        output: SchedulerOutput,
        ready: ReadyHandle,
        host: Mapping[str, Req],
    ) -> ModelRunnerOutput:
        """执行已 ready 的一步；不负责 pool.done。

        `execute_model` 和 `dummy_run` 共用本路径，区别只在前者由真实
        pool ready 驱动并在 finally 打 done，后者使用 dummy ready 且不触池。
        """

        next_tokens: Dict[str, List[int]] = {}
        next_drafts: Dict[str, List[int]] = {}

        batch = self.prepare_inputs(output, host)
        _meta = self.prepare_attn(batch, ready)
        if self.model_backend == "tiny_lm":
            next_tokens, next_drafts = self._forward_tiny(output, host, batch)
        else:
            for req_id, n in output.num_scheduled_tokens.items():
                if n <= 0:
                    continue
                req = host.get(req_id)
                if req is None:
                    continue
                if not batch.is_prompt_phase.get(req_id, False):
                    next_tokens[req_id] = [0]

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
        batch: InputBatch,
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        assert self._tiny is not None
        last_logits: Dict[str, List[float]] = {}
        drafts_prompt: Dict[str, List[int]] = {}

        for req_id in batch.req_ids:
            tokens = batch.token_ids[req_id]
            qs = batch.query_start[req_id]
            qe = batch.query_end[req_id]
            logits_rows = self._tiny.forward_query_logits(tokens, qs, qe)
            if batch.is_prompt_phase.get(req_id, False):
                # EXTEND：不产出 user token；整段 prompt 算完后挂 draft
                req = host_reqs[req_id]
                prompt_len = len(req.prompt_token_ids)
                computed = batch.num_computed_tokens[req_id]
                n = batch.num_scheduled_tokens[req_id]
                if self._drafter is not None and computed + n >= prompt_len:
                    self._drafter.post_forward(req_id, req.prompt_token_ids)
                    drafts_prompt[req_id] = self._drafter.pre_forward(req_id)
                continue
            if logits_rows:
                last_logits[req_id] = logits_rows[-1]

        # TARGET_VERIFY：走 sample 内 reject（可能无 last_logits）
        spec_map = output.scheduled_spec_decode_tokens or {}
        for req_id, draft in spec_map.items():
            if draft and req_id not in last_logits and req_id in batch.req_ids:
                last_logits[req_id] = [0.0] * self._tiny.vocab_size

        sampled, drafts = self.sample_tokens(output, host_reqs, last_logits)
        drafts.update(drafts_prompt)
        return sampled, drafts

    def clear_drafter(self, req_id: str) -> None:
        if self._drafter is not None:
            self._drafter.clear(req_id)

    def dummy_run(
        self,
        *,
        num_reqs: int = 1,
        tokens_per_req: int = 1,
        step_id: int = 0,
    ) -> ModelRunnerOutput:
        """对齐 vLLM `GPUModelRunner._dummy_run`：造假 SchedulerOutput 走生产入口。

        用于 warmup / graph capture 占位；构造 dummy host req / ready，不触发
        pool.prepare_step 或 pool.done。
        """
        num_tokens = {f"dummy-{i}": tokens_per_req for i in range(num_reqs)}
        host_reqs = {
            rid: Req(
                req_id=rid,
                model_id=self._model_id or "dummy-model",
                prompt_token_ids=list(range(tokens_per_req)),
                sampling_params=SamplingParams(max_new_tokens=1),
            )
            for rid in num_tokens
        }
        output = SchedulerOutput(
            step_id=step_id,
            forward_mode=ForwardMode.EXTEND,
            num_scheduled_tokens=num_tokens,
            total_num_scheduled_tokens=sum(num_tokens.values()),
            req_forward_modes={rid: ForwardMode.EXTEND for rid in num_tokens},
            req_num_computed_at_schedule={rid: 0 for rid in num_tokens},
            can_run_graph=True,
        )
        ready = ReadyHandle(
            step_id=step_id,
            block_table_by_req={
                rid: list(range((tokens_per_req + 7) // 8)) for rid in num_tokens
            },
            slot_mapping_by_req={
                rid: list(range(tokens_per_req)) for rid in num_tokens
            },
        )
        return self._execute_prepared(output, ready, host_reqs)
