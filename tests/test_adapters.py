"""Tests for adapter versioning storage."""

from pathlib import Path

from allm.storage import SQLiteRecordStore
from allm.trainer.adapters import AdapterStore


def test_adapter_store_versions(tmp_path: Path) -> None:
    store = SQLiteRecordStore(tmp_path / "adapters.sqlite3")
    root = tmp_path / "weights"
    adapters = AdapterStore(store, root)

    first_dir = root / "scratch1"
    first_dir.mkdir(parents=True)
    (first_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    id1 = adapters.save(
        "alpha",
        first_dir,
        base_model_id="test-model",
        samples_trained=3,
        reason="first tune",
    )
    assert id1 == "alpha-lora-0001"

    second_dir = root / "scratch2"
    second_dir.mkdir(parents=True)
    (second_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    id2 = adapters.save(
        "alpha",
        second_dir,
        base_model_id="test-model",
        samples_trained=5,
        reason="second tune",
    )
    assert id2 == "alpha-lora-0002"

    history = adapters.history("alpha")
    assert len(history) == 2
    assert adapters.latest("alpha") is not None
    assert adapters.latest("alpha").adapter_id == id2
    assert Path(history[0].path).exists()
    store.close()
