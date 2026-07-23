"""参考实现：因果 attention（纯 Python）。

生产路径由 Triton kernel 替换（`kernels/attn_triton.py`，可选依赖）。
对照 vLLM `AttentionBackend` / SGLang flashinfer 路径——此处仅验证接口与数值形状。
"""

from __future__ import annotations

import math
from typing import List


def causal_attn(
    q: List[List[float]],
    k: List[List[float]],
    v: List[List[float]],
) -> List[List[float]]:
    """q/k/v: [T, D] → out [T, D]，因果 mask。"""
    t = len(q)
    d = len(q[0]) if q else 0
    scale = 1.0 / math.sqrt(d) if d else 1.0
    out: List[List[float]] = []
    for i in range(t):
        scores = []
        for j in range(t):
            if j > i:
                scores.append(float("-inf"))
            else:
                s = sum(q[i][u] * k[j][u] for u in range(d)) * scale
                scores.append(s)
        m = max(x for x in scores if x != float("-inf"))
        exps = [math.exp(s - m) if s != float("-inf") else 0.0 for s in scores]
        z = sum(exps) or 1.0
        weights = [e / z for e in exps]
        row = [0.0] * d
        for j in range(t):
            w = weights[j]
            for u in range(d):
                row[u] += w * v[j][u]
        out.append(row)
    return out
