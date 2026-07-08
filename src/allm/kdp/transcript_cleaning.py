"""Workshop transcript preprocessing (KDP Stage 2 extension).

Kids Knowledge Seeker transcripts arrive as line-oriented ASR exports:
timestamp lines, speaker tags, and continuations split across rows.
This module groups them into speaker turns, strips broadcast noise,
and maps each turn back to character spans in the raw document.

Raw bytes in :class:`~allm.kdp.types.Document` are never modified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from allm.kdp.cleaning import clean_asr_artifacts, clean_text
from allm.kdp.types import CleanSegment, Document, SpanRef

_TIMESTAMP = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")
_HEADER = re.compile(r"^(?:transkript|transcript)\.?$", re.IGNORECASE)
_SPEAKER_PREFIX = re.compile(
    r"^(?:"
    r"(?P<name>[A-Za-z][A-Za-z\s.'-]*)\s*\((?P<id>[^)]+)\)"  # Rick Crammond (RC)
    r"|\((?P<paren_id>[^)]+)\)"                               # (MK) or (Mr Keshe)
    r")\s*:?\s+",
)
# Colon after tag is optional; reject ``Mr Keshe (MK) of the Foundation`` metadata.
_INLINE_SPEAKER = re.compile(
    r"(?:"
    r"(?<=[.!?])\s+"
    r"|(?<=^)\s*"
    r")(?:"
    r"[A-Za-z][A-Za-z\s.'-]*\((?P<name_id>[^)]+)\)"
    r"|\((?P<paren_id>[^)]+)\)"
    r")\s*:?\s+(?=[A-Z\"'(]|Yes\b|No\b|Okay|Ok\b|Yeah|Thank\b|I\b|We\b|So\b|The\b|What\b|If\b|Are\b|You\b|Good\b|Hello\b|That\b|This\b|It\b|Can\b|My\b|Now\b|Let\b|Maybe\b|Well\b|And\b|But\b|When\b|How\b|Who\b|Why\b|In\b|On\b|As\b|There\b|They\b|He\b|She\b|Yep\b|Aha\b|Exactly\b|Still\b)",
)
# Broadcast / session logistics — safe to drop (no domain knowledge).
_LOGISTICS = re.compile(
    r"^(?:"
    r"(?:welcome|good (?:evening|morning)|hello) (?:everyone|to everybody|to all)"
    r"|(?:are you there|can you hear me|can i just do a short commentary)"
    r"|(?:let me just summarize from what i remember)"
    r"|(?:thank you (?:very much|rick|keyvan|mr keshe))"
    r"|(?:okay!?|ok!?)\s*(?:keyvan|rick|mr keshe)?"
    r"|(?:this is the beginning of the \d+(?:st|nd|rd|th)? kids knowledge seekers)"
    r")",
    re.IGNORECASE,
)
_TECHNICAL = re.compile(
    r"(?:"
    r"resolving technical issue"
    r"|(?:can you see|do you see) this"
    r"|change the bandwidth"
    r"|(?:headphone|microphone|livestream|skype|pixel)"
    r"|(?:communication problems|problems here hearing|you're breaking up)"
    r"|(?:we seem to have lost|spaceship institute again|fade in and out)"
    r"|(?:carry on our conversation|while we have this interlude|go with the flow)"
    r"|(?:every time.*interrupting|try it again|could you repeat that)"
    r"|enlarge the(?: picture)?"
    r"|is it better now"
    r"|insert\.\.\.\."
    r"|technical delays"
    r")",
    re.IGNORECASE,
)
_CONNECTION = re.compile(
    r"(?:"
    r"breaking up|not online again|lost but we'?re still on planet"
    r"|expecting like, a state of evolution"
    r")",
    re.IGNORECASE,
)
_NAMED_SPEAKER = re.compile(
    r"^(?:Mr\.?\s*Keshe|Mehran\s*Keshe|Keyvan(?:\s*Davani)?|Rick\s*Crammond)$",
    re.IGNORECASE,
)


def _resolve_speaker(label: str | None) -> str | None:
    """Map ASR speaker tags and full names to canonical ids (MK, KD, RC, …)."""
    if not label:
        return None
    token = label.strip()
    if not token:
        return None
    upper = token.upper()
    if upper in {"MK", "KD", "RC", "RU", "LR", "V"}:
        return upper
    if _NAMED_SPEAKER.match(token):
        name = token.lower()
        if "keshe" in name:
            return "MK"
        if "keyvan" in name:
            return "KD"
        if "rick" in name or "crammond" in name:
            return "RC"
    if len(token) <= 4 and token.isalpha():
        return upper
    return None


@dataclass(frozen=True)
class TranscriptTurn:
    """One speaker utterance with provenance into the raw file."""

    speaker: str | None
    text: str
    span_start: int
    span_end: int


def looks_like_workshop_transcript(text: str) -> bool:
    """Heuristic: timestamp-heavy line-oriented dialogue export."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 10:
        return False
    stamps = sum(1 for line in lines if _TIMESTAMP.match(line))
    return stamps >= 5 and stamps / len(lines) >= 0.12


def _speaker_label(match: re.Match[str]) -> str | None:
    raw = match.group("id") or match.group("paren_id") or "?"
    return _resolve_speaker(raw)


def _speaker_from_inline_match(match: re.Match[str]) -> str | None:
    raw = match.group("name_id") or match.group("paren_id") or "?"
    return _resolve_speaker(raw)


def _split_inline_speakers(text: str) -> list[tuple[str | None, str]]:
    """Split text on inline speaker markers like ``...? (KD): Yes!``."""
    matches = list(_INLINE_SPEAKER.finditer(text))
    if not matches:
        match = _SPEAKER_PREFIX.match(text)
        if match:
            return [(_speaker_label(match), text[match.end() :].strip())]
        return [(None, text.strip())]

    chunks: list[tuple[str | None, str]] = []
    cursor = 0
    for index, match in enumerate(matches):
        before = text[cursor : match.start()].strip()
        speaker = _speaker_from_inline_match(match)
        utterance_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        utterance = text[match.end() : utterance_end].strip()

        if before:
            if not chunks:
                lead = _SPEAKER_PREFIX.match(before)
                if lead:
                    chunks.append((_speaker_label(lead), before[lead.end() :].strip()))
                else:
                    chunks.append((None, before))
            else:
                prev_speaker, prev_text = chunks[-1]
                chunks[-1] = (prev_speaker, f"{prev_text} {before}".strip())

        if utterance:
            chunks.append((speaker, utterance))
        cursor = utterance_end

    return chunks if chunks else [(None, text.strip())]


def _is_skippable_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _TIMESTAMP.match(stripped):
        return True
    if _HEADER.match(stripped):
        return True
    return False


def parse_transcript_turns(text: str) -> list[TranscriptTurn]:
    """Group timestamp blocks into speaker turns with raw character spans."""
    turns: list[TranscriptTurn] = []
    block_lines: list[tuple[int, str]] = []
    cursor = 0
    line_starts: list[tuple[int, str]] = []
    for line in text.splitlines(keepends=True):
        line_starts.append((cursor, line.rstrip("\n\r")))
        cursor += len(line)

    index = 0
    while index < len(line_starts):
        start_offset, raw_line = line_starts[index]
        stripped = raw_line.strip()
        if _is_skippable_line(stripped):
            if _TIMESTAMP.match(stripped) and block_lines:
                turns.extend(_flush_block(block_lines))
                block_lines = []
            index += 1
            continue
        end_offset = start_offset + len(raw_line)
        block_lines.append((start_offset, end_offset, stripped))
        index += 1
    if block_lines:
        turns.extend(_flush_block(block_lines))

    return _merge_continuations(_infer_continuation_speakers(turns))


def _flush_block(lines: list[tuple[int, int, str]]) -> list[TranscriptTurn]:
    """Convert one timestamp block into one or more turns."""
    span_start = lines[0][0]
    span_end = lines[-1][1]
    merged = " ".join(line[2] for line in lines)
    chunks = _split_inline_speakers(merged)
    if len(chunks) == 1:
        speaker, utterance = chunks[0]
        return [
            TranscriptTurn(
                speaker=speaker,
                text=utterance,
                span_start=span_start,
                span_end=span_end,
            )
        ]
    # Approximate span per chunk by proportional length (raw doc order preserved).
    turns: list[TranscriptTurn] = []
    consumed = 0
    total = sum(len(c[1]) for c in chunks) or 1
    for speaker, utterance in chunks:
        chunk_len = len(utterance)
        chunk_start = span_start + int(consumed / total * (span_end - span_start))
        consumed += chunk_len
        chunk_end = span_start + int(consumed / total * (span_end - span_start))
        turns.append(
            TranscriptTurn(
                speaker=speaker,
                text=utterance,
                span_start=chunk_start,
                span_end=max(chunk_end, chunk_start + 1),
            )
        )
    return turns


def _infer_continuation_speakers(turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
    """Assign the active speaker to untagged continuation lines (workshop 2+ ASR)."""
    active: str | None = None
    resolved: list[TranscriptTurn] = []
    for turn in turns:
        speaker = turn.speaker if turn.speaker is not None else active
        if turn.speaker is not None:
            active = turn.speaker
        resolved.append(
            TranscriptTurn(
                speaker=speaker,
                text=turn.text,
                span_start=turn.span_start,
                span_end=turn.span_end,
            )
        )
    return resolved


def _merge_continuations(turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
    """Merge consecutive turns from the same speaker into one utterance."""
    if not turns:
        return []
    merged: list[TranscriptTurn] = [turns[0]]
    for turn in turns[1:]:
        prev = merged[-1]
        same_speaker = (
            turn.speaker is not None
            and prev.speaker is not None
            and turn.speaker == prev.speaker
        )
        if same_speaker or (turn.speaker is None and prev.speaker is not None):
            merged[-1] = TranscriptTurn(
                speaker=prev.speaker,
                text=f"{prev.text} {turn.text}".strip(),
                span_start=prev.span_start,
                span_end=turn.span_end,
            )
        else:
            merged.append(turn)
    return merged


def format_turn(turn: TranscriptTurn) -> str:
    """Render a turn for downstream segmentation (speaker preserved)."""
    if turn.speaker:
        return f"{turn.speaker}: {turn.text}"
    return turn.text


def is_logistics_turn(turn: TranscriptTurn) -> bool:
    """Drop pure session chatter; keep anything with substantive length."""
    body = turn.text.strip()
    if len(body) < 20:
        return True
    if _TECHNICAL.search(body):
        return True
    if turn.speaker in {"KD", "RC", "V"} and _CONNECTION.search(body):
        return True
    if _LOGISTICS.match(body) and len(body) < 160:
        return True
    return False


def clean_transcript_document(
    document: Document,
    *,
    drop_logistics: bool = True,
) -> list[CleanSegment]:
    """Stage 2 for workshop transcripts: turns -> cleaned segments + spans."""
    segments: list[CleanSegment] = []
    for turn in parse_transcript_turns(document.text):
        if drop_logistics and is_logistics_turn(turn):
            continue
        cleaned = clean_asr_artifacts(clean_text(format_turn(turn)))
        if not cleaned:
            continue
        segments.append(
            CleanSegment(
                doc_id=document.id,
                text=cleaned,
                span=SpanRef(doc=document.id, start=turn.span_start, end=turn.span_end),
                context=document.context,
            )
        )
    return segments


def render_cleaned_transcript(document: Document, segments: list[CleanSegment]) -> str:
    """Human-readable cleaned export (one turn per paragraph)."""
    return "\n\n".join(seg.text for seg in segments)


def clean_mk_turns(document: Document) -> list[str]:
    """All Mr Keshe utterances, fully cleaned — no summarisation, no length cut."""
    turns: list[str] = []
    for turn in parse_transcript_turns(document.text):
        if turn.speaker != "MK":
            continue
        if is_logistics_turn(turn):
            continue
        cleaned = clean_asr_artifacts(clean_text(turn.text))
        if cleaned:
            turns.append(cleaned)
    return turns


def render_mk_transcript(document: Document, segments: list[CleanSegment] | None = None) -> str:
    """MK-only full teaching text for training (every cleaned MK turn kept)."""
    turns = clean_mk_turns(document)
    return "\n\n".join(turns)
