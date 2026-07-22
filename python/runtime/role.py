"""进程级角色配置（D3 最小子集；完整 schema 待补）。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WorkerRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    HYBRID = "hybrid"


@dataclass
class RoleConfig:
    role: WorkerRole = WorkerRole.HYBRID
    enable_drafter: bool = False
    enable_overlap: bool = True  # 默认开；对齐 SGLang event_loop_overlap
    max_running_reqs: int = 8  # continuous batching 上限（C1）
    # D5：prepare 补拉预算；0=同步等到齐（P3 mock）
    pull_budget_ms: int = 0
    allow_partial_hit: bool = False  # False=缺块整批失败（all-or-nothing）
    # C3：mock=P3 可复现递推；tiny_lm=纯 Python 最小因果 LM
    model_backend: str = "mock"  # mock | tiny_lm
    # arena / TP / 指标标签等留 D3
