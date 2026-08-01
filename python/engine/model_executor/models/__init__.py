from engine.model_executor.models.loader import DummyModelLoader
from engine.model_executor.models.qwen.qwen3_meta import (
    QWEN3_0_6B_CONFIG,
    QWEN3_0_6B_MODEL_ID,
    QWEN3_DUMMY_WEIGHT_NAMES,
    Qwen3Config,
)
from engine.model_executor.models.registry import (
    LoadedModel,
    ModelRegistry,
    ModelSpec,
    get_model_spec,
    load_registered_model,
    register_model_spec,
    supported_model_backends,
)
from engine.model_executor.models.tiny_lm import TinyLM

__all__ = [
    "DummyModelLoader",
    "LoadedModel",
    "ModelRegistry",
    "ModelSpec",
    "QWEN3_0_6B_CONFIG",
    "QWEN3_0_6B_MODEL_ID",
    "QWEN3_DUMMY_WEIGHT_NAMES",
    "Qwen3Config",
    "Qwen3ForCausalLM",
    "Qwen3Model",
    "TinyLM",
    "get_model_spec",
    "load_registered_model",
    "register_model_spec",
    "supported_model_backends",
]


def __getattr__(name: str) -> object:
    if name in ("Qwen3ForCausalLM", "Qwen3Model"):
        from engine.model_executor.models.qwen.qwen3 import Qwen3ForCausalLM, Qwen3Model

        return {"Qwen3ForCausalLM": Qwen3ForCausalLM, "Qwen3Model": Qwen3Model}[name]
    raise AttributeError(name)
