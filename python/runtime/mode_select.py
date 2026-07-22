"""模式选择纯函数骨架（生产权威在 Go Router；此处供节点自测 / P3 回填）。

对齐 docs/architecture/data-flow.md 决策树：
- **整段**本地命中（computed >= prompt_len 且 local_hit）→ D-direct
- 部分命中 → 混部（残差 EXTEND），不得标 D_DIRECT
- 专用 Prefill/Decode 角色 → PD
失败不设 mode-to-mode fallback。
参考:Dynamo KV-aware router；SGLang cache_aware（近似树）——我们用真命中提示。
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
    # D-direct = 前缀 KV 已在执行节点 HBM、零/极小传输；必须整段本地命中
    if prompt_len > 0 and hint.local_hit and hint.computed_tokens >= prompt_len:
        return ExecMode.D_DIRECT
    if role in (WorkerRole.PREFILL, WorkerRole.DECODE):
        return ExecMode.PD_DISAGG
    return ExecMode.COLOCATED


def should_prebuilt(hint: PrefixHint, prompt_len: int) -> bool:
    """整段前缀已在 L0 → PREBUILT（跳过 extend forward）。"""
    return bool(
        prompt_len > 0
        and (hint.prebuilt or (hint.local_hit and hint.computed_tokens >= prompt_len))
    )
