from engine.model_executor.models.loader import DummyModelLoader
from engine.model_executor.models.registry import (
    LoadedModel,
    ModelRegistry,
    ModelSpec,
    get_model_spec,
    load_hf_config,
    load_registered_model,
    register_model_spec,
    supported_model_backends,
)
from engine.model_executor.models.tiny_lm import TinyLM
from transformers import Qwen3Config

__all__ = [
    "DummyModelLoader",
    "LoadedModel",
    "ModelRegistry",
    "ModelSpec",
    "Qwen3Config",
    "Qwen3ForCausalLM",
    "Qwen3Model",
    "TinyLM",
    "get_model_spec",
    "load_hf_config",
    "load_registered_model",
    "register_model_spec",
    "supported_model_backends",
]


def __getattr__(name: str) -> object:
    if name in ("Qwen3ForCausalLM", "Qwen3Model"):
        from engine.model_executor.models.qwen.qwen3 import Qwen3ForCausalLM, Qwen3Model

        return {"Qwen3ForCausalLM": Qwen3ForCausalLM, "Qwen3Model": Qwen3Model}[name]
    raise AttributeError(name)
