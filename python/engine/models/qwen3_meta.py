"""Qwen3 metadata that is safe to import without torch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


QWEN3_0_6B_MODEL_ID = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True)
class Qwen3Config:
    model_type: str = "qwen3"
    architecture: str = "Qwen3ForCausalLM"
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 40960
    max_window_layers: int = 28
    rope_theta: int = 1000000
    torch_dtype: str = "bfloat16"
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    tie_word_embeddings: bool = True
    use_cache: bool = True
    use_sliding_window: bool = False

    @classmethod
    def from_hf_config(cls, raw: Mapping[str, object]) -> "Qwen3Config":
        arch = raw.get("architectures") or [cls.architecture]
        architecture = str(arch[0] if isinstance(arch, list) and arch else cls.architecture)
        return cls(
            model_type=str(raw.get("model_type", cls.model_type)),
            architecture=architecture,
            vocab_size=int(raw.get("vocab_size", cls.vocab_size)),
            hidden_size=int(raw.get("hidden_size", cls.hidden_size)),
            intermediate_size=int(raw.get("intermediate_size", cls.intermediate_size)),
            num_hidden_layers=int(raw.get("num_hidden_layers", cls.num_hidden_layers)),
            num_attention_heads=int(raw.get("num_attention_heads", cls.num_attention_heads)),
            num_key_value_heads=int(raw.get("num_key_value_heads", cls.num_key_value_heads)),
            head_dim=int(raw.get("head_dim", cls.head_dim)),
            max_position_embeddings=int(
                raw.get("max_position_embeddings", cls.max_position_embeddings)
            ),
            max_window_layers=int(raw.get("max_window_layers", cls.max_window_layers)),
            rope_theta=int(raw.get("rope_theta", cls.rope_theta)),
            torch_dtype=str(raw.get("torch_dtype", cls.torch_dtype)),
            rms_norm_eps=float(raw.get("rms_norm_eps", cls.rms_norm_eps)),
            attention_bias=bool(raw.get("attention_bias", cls.attention_bias)),
            attention_dropout=float(raw.get("attention_dropout", cls.attention_dropout)),
            hidden_act=str(raw.get("hidden_act", cls.hidden_act)),
            bos_token_id=int(raw.get("bos_token_id", cls.bos_token_id)),
            eos_token_id=int(raw.get("eos_token_id", cls.eos_token_id)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", cls.tie_word_embeddings)),
            use_cache=bool(raw.get("use_cache", cls.use_cache)),
            use_sliding_window=bool(raw.get("use_sliding_window", cls.use_sliding_window)),
        )

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def validate(self) -> None:
        if self.model_type != "qwen3":
            raise ValueError(f"unsupported model_type={self.model_type}")
        if self.architecture != "Qwen3ForCausalLM":
            raise ValueError(f"unsupported architecture={self.architecture}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.use_sliding_window:
            raise ValueError("Qwen3 skeleton only supports dense full attention")


QWEN3_0_6B_CONFIG = Qwen3Config()
QWEN3_DUMMY_WEIGHT_NAMES = (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.qkv_proj.weight",
    "model.layers.0.mlp.gate_up_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
)
