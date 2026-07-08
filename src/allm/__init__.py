"""ALLM — Autonomous Learning Language Model.

An experimental research platform for autonomous learning in language
models. See ``Plan.md`` (project root) for the research vision and
``docs/architecture.md`` for how the codebase maps onto it.

Package layout (one responsibility per subpackage):

- ``allm.core``      configuration, logging, plugin registry, DI container
- ``allm.storage``   versioned record storage (nothing is ever overwritten)
- ``allm.tracking``  experiment tracking (runs, params, metrics, artifacts)
- ``allm.models``    language-model interfaces and loaders
- ``allm.data``      dataset interfaces and loaders
- ``allm.cli``       project command-line interface

Later phases (teacher, students, planner, memory, knowledge, debate,
exam, trainer, collector, compression, evaluation) plug into these
interfaces; they are declared as empty subpackages so the intended
structure is visible from day one.
"""

__version__ = "0.9.0"
