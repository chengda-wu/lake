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
    # arena / TP / 指标标签等留 D3
