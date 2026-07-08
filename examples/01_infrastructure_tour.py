"""Tour of the Phase 1 infrastructure.

Runs entirely offline (echo model, temp storage). Execute with:

    python examples/01_infrastructure_tour.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from allm.core.config import load_config
from allm.core.container import Container
from allm.core.logging import get_logger, setup_logging
from allm.data.base import DatasetSpec, load_dataset
from allm.models.base import ModelSpec, load_model
from allm.storage import SQLiteRecordStore
from allm.tracking import LocalTracker


def main() -> None:
    setup_logging("INFO")
    log = get_logger("examples.tour")
    workdir = Path(tempfile.mkdtemp(prefix="allm-tour-"))
    log.info("working in %s", workdir)

    # 1. Configuration: defaults + overrides, immutable, resolved paths.
    config = load_config(overrides={"project_root": str(workdir)}).resolved()
    log.info("storage at %s", config.storage.path)

    # 2. Dependency injection: wire services once, resolve anywhere.
    container = Container()
    container.register(SQLiteRecordStore, lambda c: SQLiteRecordStore(config.storage.path))
    container.register(LocalTracker, lambda c: LocalTracker(config.tracking.root))

    # 3. Versioned storage: updates append, history survives.
    store = container.resolve(SQLiteRecordStore)
    store.put("beliefs", "gravity", {"confidence": 0.3}, reason="initial guess")
    store.put("beliefs", "gravity", {"confidence": 0.8}, reason="passed exam")
    history = store.history("beliefs", "gravity")
    log.info("gravity belief history: %s", [(r.version, r.value, r.reason) for r in history])

    # 4. Model loading through the registry (echo = deterministic mock).
    model = load_model(ModelSpec(name="tour", provider="echo", model_id="none"))
    log.info("model says: %s", model.generate("What is gravity?"))

    # 5. Dataset loading: normalised samples from JSONL.
    data_file = workdir / "tasks.jsonl"
    data_file.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"input": "2+2?", "target": "4", "topic": "math"},
                {"input": "Capital of France?", "target": "Paris", "topic": "geography"},
            ]
        ),
        encoding="utf-8",
    )
    samples = list(
        load_dataset(DatasetSpec(name="tasks", source="jsonl", location=str(data_file)))
    )
    log.info("loaded %d samples, first: %s", len(samples), samples[0].input)

    # 6. Experiment tracking: params, metrics, artifacts per run.
    tracker = container.resolve(LocalTracker)
    run = tracker.start_run("tour")
    run.log_params({"model": model.spec.name, "samples": len(samples)})
    for step, sample in enumerate(samples, start=1):
        answer = model.generate(sample.input)
        run.log_metric("answered", 1.0, step=step)
        run.log_artifact(f"answer-{step}.txt", f"Q: {sample.input}\nA: {answer}")
    run.finish()
    log.info("run recorded at %s", run.directory)

    store.close()
    print(f"\nDone. Inspect the results under {workdir}")


if __name__ == "__main__":
    main()
