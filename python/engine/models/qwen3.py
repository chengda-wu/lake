"""Qwen3 model config and runtime skeleton.

首个真实模型族先按 Qwen3-0.6B 的 Hugging Face config 固定接口；类名与
vLLM `Qwen3ForCausalLM` 对齐。本阶段经通用 `DummyModelLoader` 建立权重
加载边界和 deterministic forward 占位，不加载真实 safetensors。
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from engine.models.qwen3_meta import (
    QWEN3_0_6B_CONFIG,
    QWEN3_0_6B_MODEL_ID,
    QWEN3_DUMMY_WEIGHT_NAMES,
    Qwen3Config,
)


class Qwen3Model(nn.Module):
    """Qwen3 decoder backbone skeleton.

    vLLM 的 `Qwen3Model` 继承 `Qwen2Model` 并替换 decoder layer 类型；
    lake 先钉住 module 边界和 config，真 attention/MLP 后续接 Torch/Triton。
    """

    def __init__(self, config: Qwen3Config = QWEN3_0_6B_CONFIG) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embed_tokens = nn.Identity()
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        )
        self.norm = nn.Identity()

    def forward(
        self,
        input_ids: object | None = None,
        positions: object | None = None,
        intermediate_tensors: object | None = None,
        inputs_embeds: object | None = None,
    ) -> object:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        return self.norm(hidden_states)


class Qwen3DecoderLayer(nn.Module):
    """Dense Qwen3 decoder layer placeholder with stable module identity."""

    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.self_attn = nn.Identity()
        self.mlp = nn.Identity()
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()

    def forward(self, hidden_states: object, positions: object | None = None) -> object:
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        return self.mlp(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 causal LM skeleton.

    对齐 vLLM `Qwen3ForCausalLM`:顶层模型继承 `nn.Module`，持有
    `self.model = Qwen3Model(...)`，并暴露 `forward` / `compute_logits` /
    `load_weights(weights)`；dummy 由 loader 层处理。
    """

    def __init__(self, config: Qwen3Config = QWEN3_0_6B_CONFIG) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = self.model.embed_tokens if config.tie_word_embeddings else nn.Identity()
        self.loaded_weights: set[str] = set()
        self.loaded_dummy_weights = False

    def forward(
        self,
        input_ids: object | None = None,
        positions: object | None = None,
        intermediate_tensors: object | None = None,
        inputs_embeds: object | None = None,
    ) -> object:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: object) -> object:
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, object]]) -> set[str]:
        self.loaded_weights = {name for name, _ in weights}
        return set(self.loaded_weights)


