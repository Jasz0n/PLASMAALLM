"""The frozen wire format: the platform's build-against contract (M51).

Platform teams should not have to read our source — or track our package
version — to know what an evidence package looks like on the wire. This
module assembles a single, versioned JSON-Schema document from the
actual request/response models, so the published contract can never
silently drift from the code, yet carries its own compatibility promise
independent of both the HTTP transport and the ``allm`` version.
"""

from allm.wire.contract import WIRE_VERSION, wire_contract

__all__ = ["WIRE_VERSION", "wire_contract"]
