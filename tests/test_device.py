"""Tests for torch device resolution (no GPU required)."""

from unittest.mock import MagicMock, patch

import pytest

from allm.models.device import resolve_torch_device, resolve_torch_dtype


@patch("allm.models.device.logger")
def test_resolve_auto_picks_cuda(_log: MagicMock) -> None:
    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    with patch.dict("sys.modules", {"torch": torch}):
        assert resolve_torch_device("auto") == "cuda"


@patch("allm.models.device.logger")
def test_resolve_auto_falls_back_to_cpu(_log: MagicMock) -> None:
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    torch.backends.mps.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": torch}):
        assert resolve_torch_device("auto") == "cpu"


def test_resolve_cuda_unavailable_falls_back() -> None:
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": torch}):
        assert resolve_torch_device("cuda") == "cpu"


def test_resolve_dtype_auto_cpu() -> None:
    torch = MagicMock()
    torch.float32 = "float32"
    with patch.dict("sys.modules", {"torch": torch}):
        assert resolve_torch_dtype("auto", "cpu") == "float32"
