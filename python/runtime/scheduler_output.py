"""SchedulerOutput 信封（D1）。

外形偏 vLLM 增量 RPC；ForwardMode / DP 字段偏 SGLang；无引擎权威 block_ids。
参考:vLLM `vllm/v1/core/sched/output.py::SchedulerOutput`；
SGLang `forward_batch_info.py::ForwardMode`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ForwardMode(str, Enum):
    EXTEND = "EXTEND"
    DECODE = "DECODE"
    MIXED = "MIXED"
    IDLE = "IDLE"
    TARGET_VERIFY = "TARGET_VERIFY"
    DRAFT_EXTEND = "DRAFT_EXTEND"
    PREBUILT = "PREBUILT"


@dataclass
class SamplingParams:
    max_new_tokens: int = 16
    temperature: float = 1.0
    # 首版仅占位；完整对照见 docs/research/sampling-params.md


@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    num_computed_tokens: int = 0
    lora_id: Optional[str] = None


@dataclass
class CachedRequestData:
    req_ids: List[str] = field(default_factory=list)
    num_computed_tokens: List[int] = field(default_factory=list)
    num_output_tokens: List[int] = field(default_factory=list)
    new_token_ids: Optional[Dict[str, List[int]]] = None


@dataclass
class ReqIoSet:
    """逻辑 KV 范围；agent 据此填固定地址 block table（非物理句柄）。"""

    req_id: str
    token_start: int
    token_end: int
    pool_kind: str = "TARGET"  # TARGET | DRAFT，对齐 schema.pool_kind


@dataclass
class SchedulerOutput:
    step_id: int
    forward_mode: ForwardMode
    scheduled_new_reqs: List[NewRequestData] = field(default_factory=list)
    scheduled_cached_reqs: CachedRequestData = field(default_factory=CachedRequestData)
    num_scheduled_tokens: Dict[str, int] = field(default_factory=dict)
    total_num_scheduled_tokens: int = 0
    read_set: List[ReqIoSet] = field(default_factory=list)
    write_set: List[ReqIoSet] = field(default_factory=list)
    global_num_tokens: Optional[List[int]] = None
    can_run_graph: Optional[bool] = None
    scheduled_spec_decode_tokens: Optional[Dict[str, List[int]]] = None
    has_structured_output: bool = False
