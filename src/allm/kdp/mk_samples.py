"""Build training samples from cleaned Mr Keshe (MK) prose.

KDP knowledge units produce noisy concept names; this module reads
``cleaned/mk/*.txt`` paragraphs directly and extracts labelled Q→A
pairs from definitional and teaching sentences.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from allm.data.base import Sample
from allm.exam.grading import normalise
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.kdp.types import content_hash

SampleKind = Literal["definition", "we_call", "compact", "teaching"]
DEFAULT_KINDS: frozenset[str] = frozenset({"definition", "we_call", "compact", "teaching"})

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_LEAD = re.compile(r"^(?:so,|in fact,|but,|and,|now,|well,|okay,|yes,)\s*", re.IGNORECASE)
_DEFINITION = re.compile(
    r"^([A-Z][A-Za-z0-9 \-']{2,55}?)\s+(is|are|means|refers to|is called)\s+(.+)$",
    re.IGNORECASE,
)
_SHOULD_BE = re.compile(
    r"(?:^|,\s*)(?:a |an )?([A-Za-z][A-Za-z0-9 \-']{2,40}?),\s*to (?:a )?\w+,?\s*should be\s+(.+)$",
    re.IGNORECASE,
)
_WE_CALL = re.compile(
    r"(?P<context>[^,\.;]{8,90}),\s*we call it (?:an? )?(?P<name>[^,\.;]{3,50})",
    re.IGNORECASE,
)
_WE_CALL_SIMPLE = re.compile(
    r"\bwe call (?:it|them) (?:an? )?(?P<name>[^,\.;]{3,50})",
    re.IGNORECASE,
)
_SKIP = re.compile(
    r"(?:"
    r"can you hear|sweet dreams|see each other|go back and come again"
    r"|give us a few seconds|what are the questions|any questions\?"
    r"|all the best and thank you|do you want us to go back"
    r")",
    re.IGNORECASE,
)
_BAD_SUBJECT = re.compile(
    r"^(?:"
    r"and|but|so|yeah|yes|this|that|they|what|when|where|if|can|do|the|"
    r"he|she|it|we|you|i|oh|just|then|now|because|maybe|evolve|withholding"
    r")\b",
    re.IGNORECASE,
)
_VERB_PHRASE = re.compile(
    r"\b(?:until|when|while|because|although|before|after|if|that)\b",
    re.IGNORECASE,
)
_TEACHING_WORDS = re.compile(
    r"\b(?:plasma|magnetic|magnet|gravity|field|atom|chemistry|biology|"
    r"physics|universe|creation|orange|energy|electron|matter|proton|neutron)\b",
    re.IGNORECASE,
)
_MIN_ANSWER = 20
_MAX_ANSWER = 480
_MAX_SUBJECT_WORDS = 6


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences without breaking quoted speech."""
    parts = _SENTENCE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def normalise_sentence(sentence: str) -> str:
    """Strip discourse openers so definition patterns can match."""
    cleaned = sentence.strip()
    while True:
        next_text = _LEAD.sub("", cleaned, count=1)
        if next_text == cleaned:
            break
        cleaned = next_text.strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def is_good_subject(subject: str) -> bool:
    """Reject ASR fragments and discourse openers as fake 'concepts'."""
    subject = subject.strip().rstrip(",")
    if len(subject) < 3 or len(subject) > 60:
        return False
    if _BAD_SUBJECT.match(subject):
        return False
    if _VERB_PHRASE.search(subject):
        return False
    if subject.endswith((" which", " who", " that")):
        return False
    words = subject.split()
    if len(words) > _MAX_SUBJECT_WORDS:
        return False
    if len(words) == 1 and words[0].lower() not in {
        "plasma",
        "gravity",
        "magnetism",
        "matter",
        "physics",
        "chemistry",
        "biology",
        "energy",
        "creation",
    }:
        return False
    return True


def definition_prompt(subject: str, verb: str) -> str:
    """Short exam-friendly question for a definitional pair."""
    if verb in {"are", "refer to"}:
        return f"What are {subject}?"
    if verb == "means":
        return f"What does {subject} mean?"
    return f"What is {subject}?"


def _sample_id(source: str, prompt: str) -> str:
    return f"mk_{content_hash(source, prompt)}"


def _definition_sample(sentence: str, source: str, *, topic: str) -> Sample | None:
    normalised = normalise_sentence(sentence)
    match = _DEFINITION.match(normalised)
    if not match:
        return None
    subject, verb, rest = match.group(1).strip(), match.group(2).lower(), match.group(3).strip()
    if not is_good_subject(subject):
        return None
    answer = f"{subject} {verb} {rest}".strip().rstrip(".")
    if len(answer) < _MIN_ANSWER:
        return None
    prompt = definition_prompt(subject, verb)
    return Sample(
        id=_sample_id(source, prompt),
        input=prompt,
        target=answer[:_MAX_ANSWER],
        metadata={"topic": topic, "source": source, "sample_kind": "definition"},
    )


def _should_be_sample(sentence: str, source: str, *, topic: str) -> Sample | None:
    match = _SHOULD_BE.search(sentence)
    if not match:
        return None
    subject, rest = match.group(1).strip(), match.group(2).strip().rstrip(".")
    if not is_good_subject(subject):
        return None
    answer = f"{subject} should be {rest}".strip()
    if len(answer) < _MIN_ANSWER:
        return None
    prompt = f"What should {subject} be like, according to Mr Keshe?"
    return Sample(
        id=_sample_id(source, prompt),
        input=prompt,
        target=answer[:_MAX_ANSWER],
        metadata={"topic": topic, "source": source, "sample_kind": "definition"},
    )


def _we_call_sample(sentence: str, source: str, *, topic: str) -> Sample | None:
    match = _WE_CALL.search(sentence) or _WE_CALL_SIMPLE.search(sentence)
    if not match:
        return None
    name = match.group("name").strip().rstrip(".")
    if len(name) < 3 or len(name.split()) > 8:
        return None
    context = match.groupdict().get("context")
    if context:
        context = context.strip()
        if " that " in context.lower():
            context = context.lower().split(" that ")[-1].strip()
        else:
            words = context.split()
            context = " ".join(words[-4:])
        if len(context) >= 4 and not _BAD_SUBJECT.match(context):
            prompt = f"What do the Kids workshops call {context}?"
        else:
            prompt = f"In Kids plasma science, what is the name for {name}?"
    else:
        prompt = f"In Kids plasma science, what is {name}?"
    return Sample(
        id=_sample_id(source, prompt),
        input=prompt,
        target=sentence[:_MAX_ANSWER],
        metadata={"topic": topic, "source": source, "sample_kind": "we_call"},
    )


def _keyword_from_sentence(sentence: str) -> str | None:
    match = _TEACHING_WORDS.search(sentence)
    if not match:
        return None
    return match.group(0).lower()


def _compact_sample(sentence: str, source: str, *, topic: str) -> Sample | None:
    """Shorter Q→A for teaching facts keyed on one plasma-science term."""
    if _SKIP.search(sentence) or len(sentence) < 45:
        return None
    if sentence.strip().endswith("?"):
        return None
    keyword = _keyword_from_sentence(sentence)
    if keyword is None:
        return None
    words = sentence.split()
    hook = " ".join(words[: min(6, len(words))]).rstrip(",;:").lower()
    prompt = f"What does Mr Keshe teach about {keyword} ({hook})?"
    return Sample(
        id=_sample_id(source, prompt),
        input=prompt,
        target=sentence[:_MAX_ANSWER],
        metadata={"topic": topic, "source": source, "sample_kind": "compact"},
    )


def _teaching_sample(sentence: str, source: str, *, topic: str) -> Sample | None:
    """Legacy long-hook teaching sample (kept for ALLM_SAMPLE_KIND=teaching)."""
    if _SKIP.search(sentence) or not _TEACHING_WORDS.search(sentence):
        return None
    if len(sentence) < 45 or sentence.strip().endswith("?"):
        return None
    words = sentence.split()
    hook = " ".join(words[: min(12, len(words))]).rstrip(",;:")
    prompt = f"According to Mr Keshe's Kids workshops, explain: {hook}?"
    return Sample(
        id=_sample_id(source, prompt),
        input=prompt,
        target=sentence[:_MAX_ANSWER],
        metadata={"topic": topic, "source": source, "sample_kind": "teaching"},
    )


def paragraphs_from_mk_text(text: str) -> list[str]:
    """One MK turn per paragraph block."""
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 40]


def paragraph_to_samples(
    paragraph: str,
    source: str,
    *,
    topic: str = DEFAULT_TOPIC,
    kinds: frozenset[str] | None = None,
) -> list[Sample]:
    """Extract all usable Q→A pairs from one MK paragraph."""
    allowed = kinds or DEFAULT_KINDS
    if _SKIP.search(paragraph) and len(paragraph) < 120:
        return []
    samples: list[Sample] = []
    builders: list[tuple[str, object]] = [
        ("definition", _definition_sample),
        ("definition", _should_be_sample),
        ("we_call", _we_call_sample),
        ("compact", _compact_sample),
        ("teaching", _teaching_sample),
    ]
    for sentence in split_sentences(paragraph):
        if _SKIP.search(sentence):
            continue
        for kind, builder in builders:
            if kind not in allowed:
                continue
            sample = builder(sentence, source, topic=topic)
            if sample is not None:
                samples.append(sample)
                break
    return samples


def mk_file_to_samples(
    path: Path | str,
    *,
    topic: str = DEFAULT_TOPIC,
    kinds: frozenset[str] | None = None,
) -> list[Sample]:
    """Load one cleaned MK export and extract samples."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    samples: list[Sample] = []
    for paragraph in paragraphs_from_mk_text(text):
        samples.extend(paragraph_to_samples(paragraph, path.name, topic=topic, kinds=kinds))
    return samples


def dedupe_samples(samples: list[Sample]) -> list[Sample]:
    """Keep the richest answer when prompts normalise to the same key."""
    priority = {"definition": 0, "we_call": 1, "compact": 2, "teaching": 3}
    best: dict[str, Sample] = {}
    for sample in samples:
        key = normalise(sample.input)
        existing = best.get(key)
        if existing is None:
            best[key] = sample
            continue
        kind = (sample.metadata or {}).get("sample_kind", "teaching")
        existing_kind = (existing.metadata or {}).get("sample_kind", "teaching")
        if priority.get(kind, 9) < priority.get(existing_kind, 9):
            best[key] = sample
        elif len(sample.target or "") > len(existing.target or ""):
            best[key] = sample
    return sorted(best.values(), key=lambda s: s.id)


def filter_samples_by_kind(samples: list[Sample], kinds: frozenset[str]) -> list[Sample]:
    """Return samples whose ``sample_kind`` is in *kinds*."""
    return [
        s
        for s in samples
        if (s.metadata or {}).get("sample_kind") in kinds
    ]


def parse_sample_kinds(raw: str | None) -> frozenset[str]:
    """Parse ``ALLM_SAMPLE_KIND`` env value (comma-separated)."""
    if not raw or raw.strip().lower() in {"all", "*"}:
        return DEFAULT_KINDS
    aliases = {
        "definitions": "definition",
        "def": "definition",
        "compact": "compact",
        "teaching": "teaching",
        "exam": "definition,we_call,compact",
    }
    tokens: set[str] = set()
    for part in raw.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in aliases:
            expanded = aliases[token]
            if "," in expanded:
                tokens.update(expanded.split(","))
            else:
                tokens.add(expanded)
        else:
            tokens.add(token)
    return frozenset(tokens & set(DEFAULT_KINDS))


def mk_corpus_to_samples(
    directory: Path | str,
    *,
    pattern: str = "*.txt",
    topic: str = DEFAULT_TOPIC,
    kinds: frozenset[str] | None = None,
) -> list[Sample]:
    """Build the full training pool from ``cleaned/mk/`` exports."""
    root = Path(directory)
    all_samples: list[Sample] = []
    for path in sorted(root.glob(pattern)):
        if path.is_file():
            all_samples.extend(mk_file_to_samples(path, topic=topic, kinds=kinds))
    return dedupe_samples(all_samples)
