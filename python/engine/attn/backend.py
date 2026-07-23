"""Attention 后端选择（对照 vLLM AttentionBackend；agent 出表、runner 出其余 metadata 留 D4）。"""

from __future__ import annotations

from typing import List, Protocol

from kernels.attn_ref import causal_attn
from kernels.attn_triton import causal_attn_triton


class AttentionBackend(Protocol):
    name: str

    def forward(
        self,
        q: List[List[float]],
        k: List[List[float]],
        v: List[List[float]],
    ) -> List[List[float]]: ...


class RefAttentionBackend:
    name = "ref"

    def forward(
        self,
        q: List[List[float]],
        k: List[List[float]],
        v: List[List[float]],
    ) -> List[List[float]]:
        return causal_attn(q, k, v)


class TritonAttentionBackend:
    name = "triton"

    def forward(
        self,
        q: List[List[float]],
        k: List[List[float]],
        v: List[List[float]],
    ) -> List[List[float]]:
        return causal_attn_triton(q, k, v)


def build_attn_backend(name: str = "triton") -> AttentionBackend:
    if name == "ref":
        return RefAttentionBackend()
    return TritonAttentionBackend()
