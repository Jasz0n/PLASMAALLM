"""Official Python client for the ALLM API (Roadmap M52).

    from allm.client import AllmClient
    allm = AllmClient("https://api.example", token="…")
    allm.submit_evidence(claim="…", concept="plasma", contributor="me", outcome="supported")

Zero dependencies (stdlib only). See ``docs/client-guide.md``.
"""

from allm.client.client import (
    AllmClient,
    AllmError,
    Response,
    Transport,
    UrllibTransport,
)

__all__ = [
    "AllmClient",
    "AllmError",
    "Response",
    "Transport",
    "UrllibTransport",
]
