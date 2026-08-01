"""Model loader skeletons.

对齐 vLLM:dummy 是通用 model-loader 层能力，不属于某个模型类。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, TypeVar


TModel = TypeVar("TModel")
TConfig = TypeVar("TConfig")


class DummyModelLoader(Generic[TModel, TConfig]):
    """Generic dummy loader that constructs a model and initializes fake weights."""

    def __init__(
        self,
        model_cls: type[TModel],
        config: TConfig,
        weight_names: Iterable[str] | None = None,
    ) -> None:
        self._model_cls = model_cls
        self._config = config
        self._weight_names = tuple(weight_names) if weight_names is not None else None

    def load_model(self) -> TModel:
        model = self._model_cls(self._config)
        loaded = self.load_weights(model)
        setattr(model, "loaded_dummy_weights", True)
        setattr(model, "loaded_weights", loaded)
        return model

    def load_weights(self, model: TModel) -> set[str]:
        load_weights = getattr(model, "load_weights")
        return load_weights(self.iter_dummy_weights(model))

    def iter_dummy_weights(self, model: TModel) -> Iterable[tuple[str, object]]:
        names = self._weight_names
        if names is None:
            state_dict = getattr(model, "state_dict")
            names = tuple(state_dict().keys())
        for name in names:
            yield name, object()
