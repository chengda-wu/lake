"""SchedulerOutput 信封（D1）。

外形与调度几何偏 vLLM：`num_computed_tokens` + 本步 `num_scheduled_tokens`；
`ForwardMode` 仅为由几何派生的批标签（图/日志），不驱动分相状态机。
参考:vLLM `vllm/v1/core/sched/output.py::SchedulerOutput`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ForwardMode(str, Enum):
    """由本步 token 几何派生的批标签（非 SGLang 式执行态）。"""

    # 本步主要在算 prompt 残差（num_computed < prompt_len）
    EXTEND = "EXTEND"
    # 本步在生成（computed >= prompt_len）；含普通 decode
    DECODE = "DECODE"
    MIXED = "MIXED"
    IDLE = "IDLE"
    # 本步带 speculative verify（scheduled_spec_decode_tokens 非空）
    TARGET_VERIFY = "TARGET_VERIFY"
    DRAFT_EXTEND = "DRAFT_EXTEND"  # 预留
    # 已废弃为调度驱动：保留枚举以免旧测试/日志炸；勿再 schedule 此态
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
    # 每请求派生标签（与批级 forward_mode 一致来源：几何 / spec）
    req_forward_modes: Dict[str, ForwardMode] = field(default_factory=dict)
    # 调度瞬间的 num_computed（供 process 区分 prompt 残差 vs 生成；overlap 下 Host 可能滞后）
    req_num_computed_at_schedule: Dict[str, int] = field(default_factory=dict)
