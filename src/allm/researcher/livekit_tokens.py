"""LiveKit JWT access tokens for Researcher observer participants."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

from allm.core.logging import get_logger
from allm.researcher.livekit_types import LiveKitConfig, LiveKitCredentials, LiveStreamInfo

logger = get_logger("researcher.livekit_tokens")

DEFAULT_RESEARCHER_IDENTITY = "plasma-researcher"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def load_livekit_config() -> LiveKitConfig | None:
    """Read LiveKit credentials from environment (matches SocialServer)."""
    url = os.environ.get("LIVEKIT_URL", "").strip()
    api_key = os.environ.get("LIVEKIT_API_KEY", "").strip()
    api_secret = os.environ.get("LIVEKIT_API_SECRET", "").strip()
    if not url or not api_key or not api_secret:
        return None
    return LiveKitConfig(url=url, api_key=api_key, api_secret=api_secret)


def create_livekit_token(
    config: LiveKitConfig,
    *,
    identity: str,
    room_name: str,
    can_publish: bool = False,
    can_subscribe: bool = True,
    ttl_sec: int = 3600,
) -> str:
    """Mint a LiveKit access token compatible with livekit-server-sdk."""
    now = int(time.time())
    payload = {
        "iss": config.api_key,
        "sub": identity,
        "nbf": now,
        "exp": now + ttl_sec,
        "video": {
            "roomJoin": True,
            "room": room_name,
            "canPublish": can_publish,
            "canSubscribe": can_subscribe,
        },
    }
    header = {"alg": "HS256", "typ": "JWT"}
    segments = [
        _b64url(json.dumps(header, separators=(",", ":")).encode()),
        _b64url(json.dumps(payload, separators=(",", ":")).encode()),
    ]
    signing_input = ".".join(segments).encode()
    signature = hmac.new(config.api_secret.encode(), signing_input, hashlib.sha256).digest()
    segments.append(_b64url(signature))
    return ".".join(segments)


def credentials_for_stream(
    stream: LiveStreamInfo,
    config: LiveKitConfig,
    *,
    identity: str | None = None,
) -> LiveKitCredentials:
    """Build observer credentials for one live stream."""
    participant = identity or os.environ.get("ALLM_LIVEKIT_IDENTITY", DEFAULT_RESEARCHER_IDENTITY)
    room_name = stream.livekit_room_name or stream.stream_id
    url = stream.livekit_url or config.url
    token = create_livekit_token(
        config,
        identity=participant,
        room_name=room_name,
        can_publish=False,
        can_subscribe=True,
    )
    return LiveKitCredentials(
        stream_id=stream.stream_id,
        url=url,
        room_name=room_name,
        token=token,
        identity=participant,
    )
