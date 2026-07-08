"""Tests for Ollama vision captioning."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from allm.researcher.ollama_vision import (
    OllamaVisionClient,
    build_vision_prompt,
    encode_image_base64,
    ollama_reachable,
)
from allm.researcher.vision_caption import OllamaVisionCaptioner, StubVisionCaptioner, get_vision_captioner


def test_build_vision_prompt_includes_context() -> None:
    prompt = build_vision_prompt(
        transcript_excerpt="as you've seen in the video the plasma twists",
        tags=("blue-plasma",),
        concept_hints=("magnetical beat",),
    )
    assert "plasma" in prompt
    assert "blue-plasma" in prompt


def test_encode_image_base64(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake-jpeg-bytes")
    encoded = encode_image_base64(image)
    assert encoded


@patch("allm.researcher.ollama_vision.urllib.request.urlopen")
def test_ollama_vision_client_describe_image(mock_urlopen: MagicMock, tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake-jpeg")
    payload = json.dumps({"message": {"content": "Blue plasma between rotating magnets."}}).encode()
    mock_urlopen.return_value.__enter__.return_value.read.return_value = payload

    client = OllamaVisionClient(model_id="llava")
    caption = client.describe_image(image, "Describe this frame.")
    assert "Blue plasma" in caption
    request = mock_urlopen.call_args[0][0]
    assert request.full_url.endswith("/api/chat")
    body = json.loads(request.data.decode())
    assert body["model"] == "llava"
    assert body["messages"][0]["images"]


def test_ollama_captioner_uses_client_when_frame_exists(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake-jpeg")
    client = MagicMock()
    client.describe_image.return_value = "Magnets rotating on a pin."
    captioner = OllamaVisionCaptioner(client)
    caption = captioner.caption(
        transcript_excerpt="look at the plasma in the video",
        tags=("magnet",),
        frame_path=str(image),
    )
    assert caption.startswith("Vision:")
    client.describe_image.assert_called_once()


def test_ollama_captioner_falls_back_without_frame() -> None:
    client = MagicMock()
    captioner = OllamaVisionCaptioner(client)
    caption = captioner.caption(transcript_excerpt="plasma demo in video", tags=("plasma",))
    assert "Transcript-aligned" in caption
    client.describe_image.assert_not_called()


@patch("allm.researcher.vision_caption.ollama_reachable", return_value=False)
def test_get_vision_captioner_auto_uses_stub_when_unreachable(_mock: MagicMock) -> None:
    captioner = get_vision_captioner("auto")
    assert isinstance(captioner, StubVisionCaptioner)


@patch("allm.researcher.vision_caption.ollama_reachable", return_value=True)
def test_get_vision_captioner_auto_uses_ollama_when_reachable(_mock: MagicMock) -> None:
    captioner = get_vision_captioner("auto", ollama_model="llava")
    assert isinstance(captioner, OllamaVisionCaptioner)


@patch("allm.researcher.ollama_vision.urllib.request.urlopen")
def test_ollama_reachable_true(mock_urlopen: MagicMock) -> None:
    mock_urlopen.return_value.__enter__.return_value.status = 200
    assert ollama_reachable("http://127.0.0.1:11434") is True
