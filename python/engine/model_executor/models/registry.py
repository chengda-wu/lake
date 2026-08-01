"""Model registry for compute runners.

Mirrors vLLM's shape: model ids point to configs, configs name one or more
architectures, and the registry resolves an architecture to a lazily imported
model class.  The runner should not know concrete model classes.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

from engine.model_executor.models.qwen.qwen3_meta import (
    QWEN3_0_6B_CONFIG,
    QWEN3_0_6B_MODEL_ID,
    QWEN3_DUMMY_WEIGHT_NAMES,
)


class _BaseRegisteredModel:
    def load_model_cls(self) -> type:
        raise NotImplementedError


@dataclass(frozen=True)
class _RegisteredModel(_BaseRegisteredModel):
    model_cls: type

    def load_model_cls(self) -> type:
        return self.model_cls


@dataclass(frozen=True)
class _LazyRegisteredModel(_BaseRegisteredModel):
    module_name: str
    class_name: str

    def load_model_cls(self) -> type:
        mod = importlib.import_module(self.module_name)
        return getattr(mod, self.class_name)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    backend: str
    config: Any
    runner_attr: str = ""
    dummy_weight_names: tuple[str, ...] = ()
    load_dummy_weights: bool = False

    @property
    def architectures(self) -> list[str]:
        return list(getattr(self.config, "architectures", None) or [])


@dataclass(frozen=True)
class LoadedModel:
    model_id: str
    revision: str
    backend: str
    model: object | None = None
    runner_attr: str = ""
    load_dummy_weights: bool = False


@dataclass
class _ModelRegistry:
    models: dict[str, _BaseRegisteredModel]

    def get_supported_archs(self) -> tuple[str, ...]:
        return tuple(sorted(self.models))

    def register_model(self, model_arch: str, model_cls: type | str) -> None:
        if not isinstance(model_arch, str) or not model_arch:
            raise TypeError("model_arch must be a non-empty string")
        if isinstance(model_cls, str):
            module_name, sep, class_name = model_cls.partition(":")
            if not sep or not module_name or not class_name:
                raise ValueError("Expected model_cls string in '<module>:<class>' format")
            model = _LazyRegisteredModel(module_name, class_name)
        elif isinstance(model_cls, type):
            model = _RegisteredModel(model_cls)
        else:
            raise TypeError("model_cls must be a class or '<module>:<class>' string")
        self.models[model_arch] = model

    def resolve_model_cls(self, architectures: list[str]) -> tuple[type, str]:
        for arch in architectures:
            model = self.models.get(arch)
            if model is not None:
                return model.load_model_cls(), arch
        self._raise_for_unsupported(architectures)

    def _raise_for_unsupported(self, architectures: list[str]) -> None:
        raise NotImplementedError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {self.get_supported_archs()}"
        )


def _require_model_id(model_id: str) -> None:
    if not model_id:
        raise ValueError("model_id is required to load a model")


_NOOP_BACKENDS = {"mock", "tiny_lm"}

ModelRegistry = _ModelRegistry(models={})
ModelRegistry.register_model(
    "Qwen3ForCausalLM",
    "engine.model_executor.models.qwen.qwen3:Qwen3ForCausalLM",
)

_MODEL_SPECS: dict[str, ModelSpec] = {
    QWEN3_0_6B_MODEL_ID: ModelSpec(
        model_id=QWEN3_0_6B_MODEL_ID,
        backend="qwen3",
        config=QWEN3_0_6B_CONFIG,
        runner_attr="_qwen3",
        dummy_weight_names=QWEN3_DUMMY_WEIGHT_NAMES,
        load_dummy_weights=True,
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return _MODEL_SPECS[model_id]
    except KeyError as e:
        raise NotImplementedError(f"unsupported model_id={model_id!r}") from e


def load_registered_model(
    *,
    backend: str,
    model_id: str,
    revision: str = "",
    config_override: Any | None = None,
) -> LoadedModel:
    _require_model_id(model_id)
    if backend in _NOOP_BACKENDS:
        return LoadedModel(model_id=model_id, revision=revision, backend=backend)

    spec = get_model_spec(model_id)
    if spec.backend != backend:
        raise NotImplementedError(
            f"model_id={model_id!r} is registered for backend={spec.backend!r}, "
            f"not backend={backend!r}"
        )

    config = config_override or spec.config
    architectures = list(getattr(config, "architectures", None) or spec.architectures)
    model_cls, _ = ModelRegistry.resolve_model_cls(architectures)

    from engine.model_executor.models.loader import DummyModelLoader

    model = DummyModelLoader(
        model_cls,
        config,
        spec.dummy_weight_names,
    ).load_model()
    return LoadedModel(
        model_id=spec.model_id,
        revision=revision,
        backend=spec.backend,
        model=model,
        runner_attr=spec.runner_attr,
        load_dummy_weights=spec.load_dummy_weights,
    )


def register_model_spec(spec: ModelSpec) -> None:
    _require_model_id(spec.model_id)
    _MODEL_SPECS[spec.model_id] = spec


def supported_model_backends() -> tuple[str, ...]:
    return tuple(sorted({spec.backend for spec in _MODEL_SPECS.values()} | _NOOP_BACKENDS))
