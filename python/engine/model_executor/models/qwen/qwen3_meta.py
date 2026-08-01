"""Qwen3 metadata backed by Hugging Face Transformers config."""

from __future__ import annotations

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


QWEN3_0_6B_MODEL_ID = "Qwen/Qwen3-0.6B"


QWEN3_0_6B_CONFIG = Qwen3Config(
    architectures=["Qwen3ForCausalLM"],
    vocab_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    max_position_embeddings=40960,
    max_window_layers=28,
    rope_parameters={"rope_type": "default", "rope_theta": 1000000},
    dtype="bfloat16",
    rms_norm_eps=1e-6,
    attention_bias=False,
    attention_dropout=0.0,
    hidden_act="silu",
    bos_token_id=151643,
    eos_token_id=151645,
    tie_word_embeddings=True,
    use_cache=True,
    use_sliding_window=False,
)
QWEN3_DUMMY_WEIGHT_NAMES = (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.qkv_proj.weight",
    "model.layers.0.mlp.gate_up_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
)
