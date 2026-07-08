"""LLM teaching digest — extract what MK taught from cleaned workshops.

Uses a stronger model (default: local Ollama 14b) to read Mr Keshe's
turns and produce structured learning notes faithful to the transcript.
Long sessions are processed in chunks, then merged in a final pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from allm.core.logging import get_logger
from allm.models.base import GenerationParams, LanguageModel

logger = get_logger("kdp.teaching_digest")

_TURN = re.compile(r"^(?P<speaker>[A-Z]{1,4}):\s*(?P<body>.+)$", re.MULTILINE)
_SECTION = re.compile(r"^(?P<name>[A-Z_]+):\s*(?P<body>.*)$", re.MULTILINE)
_CHUNK_CHARS = 5500


@dataclass(frozen=True)
class TeachingDigest:
    """Structured learning notes for one workshop."""

    workshop: str
    title: str
    summary: str
    key_concepts: tuple[str, ...]
    facts_for_kids: tuple[str, ...]
    misconceptions_addressed: tuple[str, ...]
    demos_and_analogies: tuple[str, ...]
    follow_up_questions: tuple[str, ...]
    raw_response: str


def extract_speaker_turns(text: str, speaker: str = "MK") -> list[str]:
    """Pull all utterances for one speaker from cleaned transcript export."""
    turns: list[str] = []
    for match in _TURN.finditer(text):
        if match.group("speaker") == speaker:
            body = match.group("body").strip()
            if len(body) >= 40:
                turns.append(body)
    return turns


def chunk_turns(turns: list[str], max_chars: int = _CHUNK_CHARS) -> list[list[str]]:
    """Split long MK sessions into model-sized chunks (chronological)."""
    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for turn in turns:
        if current and size + len(turn) > max_chars:
            chunks.append(current)
            current = []
            size = 0
        current.append(turn)
        size += len(turn)
    if current:
        chunks.append(current)
    return chunks


def build_merge_prompt(workshop: str, partials: list[str]) -> str:
    """Combine chunk-level digests into one workshop summary."""
    body = "\n\n---\n\n".join(f"Part {i + 1}:\n{p}" for i, p in enumerate(partials))
    return f"""You merge partial teaching notes from one Kids Knowledge Seekers workshop.

Workshop: {workshop}

Partial notes (same session, chronological):
{body}

Merge into one curriculum summary. Deduplicate bullets; keep MK's teaching faithful.

Reply with exactly these sections (use bullet lines starting with "- " under each list):

TITLE: one short session title
SUMMARY: 2-4 sentences on what MK taught overall
KEY_CONCEPTS:
- ...
FACTS_FOR_KIDS:
- ...
MISCONCEPTIONS_ADDRESSED:
- ...
DEMOS_AND_ANALOGIES:
- ...
FOLLOW_UP_QUESTIONS:
- ...
"""


def build_digest_prompt(workshop: str, mk_turns: list[str]) -> str:
    """Prompt for the teaching-extraction model."""
    joined = "\n\n".join(f"- {turn}" for turn in mk_turns)
    return f"""You analyze Kids Knowledge Seekers workshops for a children's science curriculum.

Workshop: {workshop}

Below is everything Mr Keshe (MK) said in this session (already cleaned; no timestamps).
Extract what children should learn. Stay faithful to MK's words — do not invent physics.

MK's teaching:
{joined}

Reply with exactly these sections (use bullet lines starting with "- " under each list):

TITLE: one short session title
SUMMARY: 2-3 sentences on what MK taught
KEY_CONCEPTS:
- ...
FACTS_FOR_KIDS:
- ...
MISCONCEPTIONS_ADDRESSED:
- ...
DEMOS_AND_ANALOGIES:
- ...
FOLLOW_UP_QUESTIONS:
- ...
"""


def _bullets(block: str) -> tuple[str, ...]:
    items: list[str] = []
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("- "):
            items.append(line[2:].strip())
        elif line.startswith("* "):
            items.append(line[2:].strip())
    return tuple(items)


def _normalize_response(text: str) -> str:
    """Accept plain or markdown-bold section headers from the model."""
    text = re.sub(r"\*\*([A-Z_]+):\*\*\s*", r"\1: ", text)
    text = re.sub(r"\*\*", "", text)
    return text.strip()


def parse_digest_response(workshop: str, text: str) -> TeachingDigest:
    """Parse the structured model reply into a :class:`TeachingDigest`."""
    text = _normalize_response(text)
    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        match = _SECTION.match(line)
        if match and match.group("name") in {
            "TITLE",
            "SUMMARY",
            "KEY_CONCEPTS",
            "FACTS_FOR_KIDS",
            "MISCONCEPTIONS_ADDRESSED",
            "DEMOS_AND_ANALOGIES",
            "FOLLOW_UP_QUESTIONS",
        }:
            if current is not None:
                sections[current] = "\n".join(buffer).strip()
            current = match.group("name")
            buffer = [match.group("body").strip()] if match.group("body").strip() else []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        sections[current] = "\n".join(buffer).strip()

    return TeachingDigest(
        workshop=workshop,
        title=sections.get("TITLE", workshop),
        summary=sections.get("SUMMARY", ""),
        key_concepts=_bullets(sections.get("KEY_CONCEPTS", "")),
        facts_for_kids=_bullets(sections.get("FACTS_FOR_KIDS", "")),
        misconceptions_addressed=_bullets(sections.get("MISCONCEPTIONS_ADDRESSED", "")),
        demos_and_analogies=_bullets(sections.get("DEMOS_AND_ANALOGIES", "")),
        follow_up_questions=_bullets(sections.get("FOLLOW_UP_QUESTIONS", "")),
        raw_response=text,
    )


def digest_mk_teaching(
    workshop: str,
    cleaned_text: str,
    model: LanguageModel,
    *,
    generation: GenerationParams | None = None,
) -> TeachingDigest:
    """Send MK's turns to a language model and return structured notes."""
    mk_turns = extract_speaker_turns(cleaned_text, speaker="MK")
    if not mk_turns:
        return TeachingDigest(
            workshop=workshop,
            title=workshop,
            summary="No MK turns found in cleaned transcript.",
            key_concepts=(),
            facts_for_kids=(),
            misconceptions_addressed=(),
            demos_and_analogies=(),
            follow_up_questions=(),
            raw_response="",
        )
    params = generation or GenerationParams(max_new_tokens=1200, temperature=0.1, top_p=1.0)
    chunks = chunk_turns(mk_turns)
    logger.info("%s: digesting %d MK turn(s) in %d chunk(s)", workshop, len(mk_turns), len(chunks))

    partial_texts: list[str] = []
    for index, chunk in enumerate(chunks):
        prompt = build_digest_prompt(f"{workshop} (part {index + 1}/{len(chunks)})", chunk)
        partial_texts.append(model.generate(prompt, params))

    if len(partial_texts) == 1:
        response = partial_texts[0]
    else:
        merge_prompt = build_merge_prompt(workshop, partial_texts)
        response = model.generate(merge_prompt, params)

    return parse_digest_response(workshop, response)


def render_digest(digest: TeachingDigest) -> str:
    """Human-readable digest for review or training pipelines."""

    def block(title: str, items: tuple[str, ...]) -> str:
        if not items:
            return ""
        lines = "\n".join(f"- {item}" for item in items)
        return f"\n{title}\n{lines}\n"

    return (
        f"# {digest.title}\n\n"
        f"Workshop: {digest.workshop}\n\n"
        f"## Summary\n{digest.summary}\n"
        f"{block('## Key concepts', digest.key_concepts)}"
        f"{block('## Facts for kids', digest.facts_for_kids)}"
        f"{block('## Misconceptions addressed', digest.misconceptions_addressed)}"
        f"{block('## Demos and analogies', digest.demos_and_analogies)}"
        f"{block('## Follow-up questions', digest.follow_up_questions)}"
    ).strip() + "\n"
