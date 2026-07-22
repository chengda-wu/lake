"""模式选择纯函数骨架（生产权威在 Go Router；此处供节点自测 / P3 回填）。

对齐 docs/architecture/data-flow.md 决策树：本地命中 → D-direct；
专用 Prefill/Decode 角色 → PD；否则混部。失败不设 mode-to-mode fallback。
参考:Dynamo KV-aware router overlap；SGLang cache_aware（近似树）——我们用真命中提示。
"""

from __future__ import annotations

from runtime.exec_mode import ExecMode
from runtime.prefix_hint import PrefixHint
from runtime.role import WorkerRole


def select_exec_mode(
    hint: PrefixHint,
    *,
    prompt_len: int,
    role: WorkerRole = WorkerRole.HYBRID,
) -> ExecMode:
    if hint.local_hit and hint.computed_tokens > 0:
        return ExecMode.D_DIRECT
    if role in (WorkerRole.PREFILL, WorkerRole.DECODE):
        return ExecMode.PD_DISAGG
    _ = prompt_len  # 阈值树留 P7
    return ExecMode.COLOCATED


def should_prebuilt(hint: PrefixHint, prompt_len: int) -> bool:
    """整段前缀已在 L0 → PREBUILT（跳过 extend forward）。"""
    return bool(hint.prebuilt or (hint.local_hit and hint.computed_tokens >= prompt_len > 0))
