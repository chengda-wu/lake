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
        weight_names: Iterable[str],
    ) -> None:
        self._model_cls = model_cls
        self._config = config
        self._weight_names = tuple(weight_names)

    def load_model(self) -> TModel:
        model = self._model_cls(self._config)
        loaded = self.load_weights(model)
        setattr(model, "loaded_dummy_weights", True)
        setattr(model, "loaded_weights", loaded)
        return model

    def load_weights(self, model: TModel) -> set[str]:
        load_weights = getattr(model, "load_weights")
        return load_weights(self.iter_dummy_weights())

    def iter_dummy_weights(self) -> Iterable[tuple[str, object]]:
        for name in self._weight_names:
            yield name, object()
