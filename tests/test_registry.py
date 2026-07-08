"""Tests for allm.core.registry."""

import pytest

from allm.core.registry import Registry, RegistryError


def test_register_and_get() -> None:
    registry: Registry[type] = Registry("widget")

    @registry.register("basic")
    class Basic: ...

    assert registry.get("basic") is Basic
    assert "basic" in registry
    assert registry.names() == ["basic"]


def test_duplicate_name_rejected() -> None:
    registry: Registry[str] = Registry("widget")
    registry.add("a", "first")
    with pytest.raises(RegistryError, match="already registered"):
        registry.add("a", "second")


def test_unknown_name_lists_alternatives() -> None:
    registry: Registry[str] = Registry("widget")
    registry.add("alpha", "a")
    with pytest.raises(RegistryError, match="alpha"):
        registry.get("beta")
