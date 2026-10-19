"""Load and resolve model catalog identifiers and source files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from TokenSim.errors import ConfigurationError
from TokenSim.hardware._yaml import load_yaml_mapping

if TYPE_CHECKING:
    from TokenSim.config.model_config import ModelSpec


class ModelCatalog:
    def __init__(self, models: Iterable[ModelSpec] = ()) -> None:
        self._models: dict[str, ModelSpec] = {}
        self._index: dict[str, str] = {}
        for model in models:
            self.add(model)

    def add(self, model: ModelSpec) -> None:
        if model.model_id in self._models:
            raise ConfigurationError(f"duplicate model_id {model.model_id!r}")
        self._models[model.model_id] = model
        for name in model.names:
            key = _alias_key(name)
            existing = self._index.get(key)
            if existing is not None and existing != model.model_id:
                raise ConfigurationError(
                    f"model alias {name!r} is claimed by both {existing!r} and {model.model_id!r}"
                )
            self._index[key] = model.model_id

    @classmethod
    def load(cls, directory: str | Path) -> "ModelCatalog":
        from TokenSim.config.model_config import ModelSpec

        directory = Path(directory)
        if not directory.is_dir():
            raise ConfigurationError(f"model catalog directory not found: {directory}")
        catalog = cls()
        for path in sorted(directory.glob("*.yaml")):
            catalog.add(
                ModelSpec.from_mapping(load_yaml_mapping(path), context=str(path))
            )
        if not catalog:
            raise ConfigurationError(f"model catalog {directory} has no *.yaml entries")
        return catalog

    def resolve(self, source: str | Path) -> ModelSpec:
        from TokenSim.config.model_config import ModelSpec

        path = Path(source).expanduser()
        if path.is_file():
            return ModelSpec.from_mapping(load_yaml_mapping(path), context=str(path))
        if path.suffix.lower() in {".yaml", ".yml"}:
            raise ConfigurationError(f"model configuration does not exist: {path}")
        return self.get(str(source))

    def get(self, name: str) -> ModelSpec:
        model_id = self._index.get(_alias_key(name))
        if model_id is None:
            raise ConfigurationError(
                f"unknown model {name!r}; known models: {sorted(self._models)}"
            )
        return self._models[model_id]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and _alias_key(name) in self._index

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self):
        return iter(self._models.values())


def _alias_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())
