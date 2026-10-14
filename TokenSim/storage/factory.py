"""Explicit registration and lazy loading of configured storage backends."""

from copy import deepcopy
import importlib
import json
from typing import Callable

from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError
from .config import BackendSpec
from .interfaces import StorageBackend


class StorageBackendFactory:
    _registry: dict[str, Callable[..., StorageBackend]] = {}

    @classmethod
    def register_backend(
        cls, name: str, builder: Callable[..., StorageBackend]
    ) -> None:
        if not isinstance(name, str) or not name.strip() or ":" in name:
            raise ConfigurationError(
                "registered storage backend names must be nonempty aliases"
            )
        if name == "ssd" or name in cls._registry:
            raise ConfigurationError(f"storage backend {name!r} is already registered")
        if not callable(builder):
            raise ConfigurationError("storage backend builder must be callable")
        cls._registry[name] = builder

    @classmethod
    def create(cls, spec: BackendSpec) -> StorageBackend:
        builder = cls._builder(spec.backend)
        try:
            backend = builder(deepcopy(spec.config))
            integer(backend.capacity_bytes, "storage backend capacity_bytes")
            if not callable(backend.estimate_io) or not callable(backend.describe):
                raise ConfigurationError(
                    "storage backend must implement estimate_io and describe"
                )
            description = backend.describe()
            if not isinstance(description, dict):
                raise ConfigurationError(
                    "storage backend describe must return an object"
                )
            json.dumps(description, allow_nan=False, sort_keys=True)
        except (
            TypeError,
            ValueError,
            KeyError,
            AttributeError,
            OverflowError,
        ) as error:
            raise ConfigurationError(
                f"invalid storage backend {spec.backend!r}: {error}"
            ) from error
        return backend

    @classmethod
    def _builder(cls, reference):
        if reference in cls._registry:
            return cls._registry[reference]
        if reference == "ssd":
            from .ssd import SSDBackend

            return SSDBackend
        module_name, separator, symbol = reference.partition(":")
        if (
            not separator
            or not all(part.isidentifier() for part in module_name.split("."))
            or not symbol.isidentifier()
        ):
            raise ConfigurationError(
                f"unknown storage backend {reference!r}; use a registered alias or module:class"
            )
        try:
            builder = getattr(importlib.import_module(module_name), symbol)
        except (ImportError, AttributeError) as error:
            raise ConfigurationError(
                f"cannot load storage backend {reference!r}; install its package and check the reference: {error}"
            ) from error
        if not callable(builder):
            raise ConfigurationError(f"storage backend {reference!r} is not callable")
        return builder
