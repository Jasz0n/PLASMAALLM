"""Plugin registry.

Design decisions
----------------
- A :class:`Registry` maps string names to factories (classes or
  functions). Components look up implementations by name taken from
  configuration, which is what makes every component replaceable
  without touching call sites.
- Third-party packages can contribute implementations through the
  ``allm.plugins`` entry-point group; :func:`load_entrypoint_plugins`
  imports them on demand. Nothing is imported implicitly.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable, Generic, Iterable, TypeVar

T = TypeVar("T")

ENTRYPOINT_GROUP = "allm.plugins"


class RegistryError(KeyError):
    """Raised for unknown or duplicate registry entries."""


class Registry(Generic[T]):
    """A named collection of interchangeable factories.

    Example::

        loaders: Registry[type[ModelLoader]] = Registry("model_loader")

        @loaders.register("huggingface")
        class HFModelLoader: ...

        loader_cls = loaders.get("huggingface")
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    @property
    def kind(self) -> str:
        return self._kind

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator registering ``name`` -> decorated object."""

        def decorator(obj: T) -> T:
            self.add(name, obj)
            return obj

        return decorator

    def add(self, name: str, obj: T) -> None:
        if name in self._items:
            raise RegistryError(f"{self._kind} {name!r} is already registered")
        self._items[name] = obj

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<none>"
            raise RegistryError(
                f"unknown {self._kind} {name!r}; registered: {known}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self) -> Iterable[str]:
        return iter(sorted(self._items))


def load_entrypoint_plugins(group: str = ENTRYPOINT_GROUP) -> list[str]:
    """Import every module advertised under ``group`` and return their names.

    Importing a plugin module is expected to register implementations
    into the relevant registries as a side effect of import (via the
    :meth:`Registry.register` decorator).
    """
    loaded: list[str] = []
    for ep in entry_points(group=group):
        ep.load()
        loaded.append(ep.name)
    return loaded
