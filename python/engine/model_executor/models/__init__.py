from engine.model_executor.models.loader import DummyModelLoader
from engine.model_executor.models.registry import (
    LoadedModel,
    ModelRegistry,
    load_hf_config,
    load_registered_model,
)
from transformers import Qwen3Config

__all__ = [
    "DummyModelLoader",
    "LoadedModel",
    "ModelRegistry",
    "Qwen3Config",
    "Qwen3ForCausalLM",
    "Qwen3Model",
    "load_hf_config",
    "load_registered_model",
]


def __getattr__(name: str) -> object:
    if name in ("Qwen3ForCausalLM", "Qwen3Model"):
        from engine.model_executor.models.qwen.qwen3 import Qwen3ForCausalLM, Qwen3Model

        return {"Qwen3ForCausalLM": Qwen3ForCausalLM, "Qwen3Model": Qwen3Model}[name]
    raise AttributeError(name)
