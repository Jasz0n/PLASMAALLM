"""Tests for allm.core.container."""

import pytest

from allm.core.container import Container, ContainerError


class Service:
    def __init__(self, name: str = "svc") -> None:
        self.name = name


def test_singleton_is_cached() -> None:
    container = Container()
    container.register(Service, lambda c: Service())
    assert container.resolve(Service) is container.resolve(Service)


def test_transient_builds_new_instances() -> None:
    container = Container()
    container.register(Service, lambda c: Service(), singleton=False)
    assert container.resolve(Service) is not container.resolve(Service)


def test_factory_can_resolve_dependencies() -> None:
    container = Container()
    container.register("name", lambda c: "wired")
    container.register(Service, lambda c: Service(c.resolve("name")))
    assert container.resolve(Service).name == "wired"


def test_register_instance() -> None:
    container = Container()
    instance = Service()
    container.register_instance(Service, instance)
    assert container.resolve(Service) is instance


def test_unknown_key_raises() -> None:
    with pytest.raises(ContainerError):
        Container().resolve(Service)


def test_duplicate_registration_rejected() -> None:
    container = Container()
    container.register(Service, lambda c: Service())
    with pytest.raises(ContainerError):
        container.register(Service, lambda c: Service())
