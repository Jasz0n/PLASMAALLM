"""Dependency-injection container.

Design decisions
----------------
- Deliberately minimal (~60 lines): services are registered under a
  key (usually the interface/protocol type) with a factory; the
  container resolves them lazily and caches singletons.
- Factories receive the container so they can resolve their own
  dependencies — composition happens in one place (the composition
  root), components never construct their collaborators.
- No decorators, no scanning, no magic: explicit wiring is easier to
  reason about in a research codebase where components are swapped
  constantly.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")

Factory = Callable[["Container"], Any]


class ContainerError(KeyError):
    """Raised when a service key is unknown or already bound."""


class Container:
    """Maps keys (types or strings) to lazily-constructed services."""

    def __init__(self) -> None:
        self._factories: dict[Any, Factory] = {}
        self._singleton_keys: set[Any] = set()
        self._instances: dict[Any, Any] = {}

    def register(self, key: type[T] | str, factory: Factory, *, singleton: bool = True) -> None:
        """Bind ``key`` to ``factory``.

        Args:
            key: Interface type or string name to resolve later.
            factory: Callable taking the container, returning the service.
            singleton: Cache the first instance (default) or build a new
                one per :meth:`resolve` call.
        """
        if key in self._factories:
            raise ContainerError(f"service {key!r} is already registered")
        self._factories[key] = factory
        if singleton:
            self._singleton_keys.add(key)

    def register_instance(self, key: type[T] | str, instance: T) -> None:
        """Bind an already-constructed object (always a singleton)."""
        self.register(key, lambda _: instance)

    def resolve(self, key: type[T] | str) -> T:
        """Return the service bound to ``key``, constructing it if needed."""
        if key in self._instances:
            return self._instances[key]
        try:
            factory = self._factories[key]
        except KeyError:
            raise ContainerError(f"no service registered for {key!r}") from None
        instance = factory(self)
        if key in self._singleton_keys:
            self._instances[key] = instance
        return instance

    def __contains__(self, key: Any) -> bool:
        return key in self._factories
