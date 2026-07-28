from engine.models.loader import DummyModelLoader
from engine.models.qwen3_meta import (
    QWEN3_0_6B_CONFIG,
    QWEN3_0_6B_MODEL_ID,
    QWEN3_DUMMY_WEIGHT_NAMES,
    Qwen3Config,
)
from engine.models.tiny_lm import TinyLM

__all__ = [
    "DummyModelLoader",
    "QWEN3_0_6B_CONFIG",
    "QWEN3_0_6B_MODEL_ID",
    "QWEN3_DUMMY_WEIGHT_NAMES",
    "Qwen3Config",
    "Qwen3ForCausalLM",
    "Qwen3Model",
    "TinyLM",
]


def __getattr__(name: str) -> object:
    if name in ("Qwen3ForCausalLM", "Qwen3Model"):
        from engine.models.qwen3 import Qwen3ForCausalLM, Qwen3Model

        return {"Qwen3ForCausalLM": Qwen3ForCausalLM, "Qwen3Model": Qwen3Model}[name]
    raise AttributeError(name)
