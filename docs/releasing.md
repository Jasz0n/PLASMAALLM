# Releasing ALLM

The package is build-ready; publishing is a human act (same rule as
everywhere: nothing leaves without a person behind it).

```bash
# 1. Bump the version in BOTH places (a test will catch drift via the API):
#    pyproject.toml  [project] version
#    src/allm/__init__.py  __version__

# 2. Refresh the published wire contract:
PYTHONPATH=src python3 - <<'PY'
import json, pathlib, tempfile
from allm.api.app import create_app
spec = create_app(pathlib.Path(tempfile.mkdtemp()) / "x.sqlite3").openapi()
pathlib.Path("docs/openapi.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
PY

# 3. Verify and build:
PYTHONPATH=src python3 -m pytest -q
python3 -m build --sdist --wheel .

# 4. Publish (needs a PyPI account + token):
python3 -m pip install twine
python3 -m twine upload dist/allm-<version>*
```

`docs/openapi.json` is the frozen API contract platform teams build
against; `tests/test_api_security.py::test_published_openapi_contract_is_current`
fails CI when it drifts from the code.
