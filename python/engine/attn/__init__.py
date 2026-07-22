"""Attention 后端边界（D4 初版）：metadata + 后端选择。"""

from engine.attn.backend import AttentionBackend, RefAttentionBackend, build_attn_backend

__all__ = ["AttentionBackend", "RefAttentionBackend", "build_attn_backend"]
