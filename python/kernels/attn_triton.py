"""Triton attention 入口（可选依赖）。

未安装 triton 时回退 `attn_ref`。真正 GPU kernel 留后续迭代。
"""

from __future__ import annotations

from typing import List

from kernels.attn_ref import causal_attn


def causal_attn_triton(
    q: List[List[float]],
    k: List[List[float]],
    v: List[List[float]],
) -> List[List[float]]:
    try:
        import triton  # noqa: F401
    except ImportError:
        return causal_attn(q, k, v)
    # C3：接口就位；数值仍走参考实现（避免无 GPU CI 挂）
    return causal_attn(q, k, v)
