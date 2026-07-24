"""进程级角色配置（D3 最小子集；完整 schema 待补）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class WorkerRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    HYBRID = "hybrid"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass
class RoleConfig:
    role: WorkerRole = WorkerRole.HYBRID
    enable_drafter: bool = False
    num_draft_tokens: int = 2  # C4：MTP 宽度
    enable_overlap: bool = True  # 默认开；对齐 SGLang event_loop_overlap
    max_running_reqs: int = 8  # continuous batching 上限（C1）
    # D5：prepare 补拉预算；0=同步等到齐（P3 mock）
    pull_budget_ms: int = 0
    allow_partial_hit: bool = False  # False=缺块整批失败（all-or-nothing）
    # C3：mock=P3 可复现递推；tiny_lm=纯 Python 最小因果 LM
    model_backend: str = "mock"  # mock | tiny_lm
    # arena / TP / 指标标签等留 D3

    @classmethod
    def from_env(cls) -> "RoleConfig":
        """C6/D3 最小：从环境变量读启动配置。

        LAKE_WORKER_ROLE=prefill|decode|hybrid
        LAKE_MODEL_BACKEND=mock|tiny_lm
        LAKE_ENABLE_DRAFTER / LAKE_ENABLE_OVERLAP / LAKE_ALLOW_PARTIAL_HIT
        LAKE_NUM_DRAFT_TOKENS / LAKE_MAX_RUNNING_REQS / LAKE_PULL_BUDGET_MS
        """
        role_raw = os.environ.get("LAKE_WORKER_ROLE", "hybrid").strip().lower()
        try:
            role = WorkerRole(role_raw)
        except ValueError:
            role = WorkerRole.HYBRID
        backend = os.environ.get("LAKE_MODEL_BACKEND", "mock").strip().lower() or "mock"
        return cls(
            role=role,
            model_backend=backend,
            enable_drafter=_env_bool("LAKE_ENABLE_DRAFTER", False),
            enable_overlap=_env_bool("LAKE_ENABLE_OVERLAP", True),
            allow_partial_hit=_env_bool("LAKE_ALLOW_PARTIAL_HIT", False),
            num_draft_tokens=_env_int("LAKE_NUM_DRAFT_TOKENS", 2),
            max_running_reqs=_env_int("LAKE_MAX_RUNNING_REQS", 8),
            pull_budget_ms=_env_int("LAKE_PULL_BUDGET_MS", 0),
        )