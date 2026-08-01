"""Attention 后端边界（D4/C8）：metadata + 后端选择。"""

from engine.model_executor.layers.attentions import (
    AttentionBackend,
    AttentionMetadata,
    RefAttentionBackend,
    TorchAttentionBackend,
    build_attn_backend,
    build_attn_metadata,
)

__all__ = [
    "AttentionBackend",
    "AttentionMetadata",
    "RefAttentionBackend",
    "TorchAttentionBackend",
    "build_attn_backend",
    "build_attn_metadata",
]
