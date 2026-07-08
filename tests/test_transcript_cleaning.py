"""Tests for workshop transcript cleaning (Kids Knowledge Seekers corpus)."""

from pathlib import Path

import pytest

from allm.kdp import DocumentStore, KDPipeline
from allm.kdp.transcript_cleaning import (
    clean_transcript_document,
    looks_like_workshop_transcript,
    parse_transcript_turns,
    render_cleaned_transcript,
    render_mk_transcript,
)

ROOT = Path(__file__).resolve().parents[1]
KIDS_DIR = ROOT / "transcripts" / "Kids"

SAMPLE = """Transkript


0:09
Rick Crammond (RC): Welcome everyone to the 1st Kids Knowledge Seekers Workshop.
0:17
My name is Rick Crammond and I will co-host this weekly series with Keyvan Davani (KD),
0:23
and Keyvan is organizing the Kids workshops, and he will make a short introduction here.
0:30
Then, we will hear from Mr Keshe (MK), who will explain the workings of the universe
0:35
in a way that the child in all of us can understand.
0:45
Keyvan, can you continue? (KD): Yes! Thank you Rick.
0:51
Again, my name is Keyvan Davani... when I was a kid, approximately six or seven years old,
0:58
I remember asking many, many, let's say creative questions to my dad,
1:04
and at that time, I could not get, so I couldn't get any answers which I was searching for.
1:10
The whole reason for this Kids workshop is to gain some understanding,
1:18
start gaining understanding for all those questions which many kids might have,
1:26
in, you know, when they are young, or going to school, learning physics, biology, chemistry,
1:32
and... and I guess, Mr Keshe, when he was a kid, he ...
1:40
most probably asked himself the same question which I had, you know, about the universe,
1:45
about our galaxies, our planets, our stars, the Moon, what are we made of,
1:52
you know, planet Earth, so, maybe, you know,
1:58
all those things that we learned in school might not be totally wrong, but since I've been following Mr Keshe, reading his books, for a few years,
2:10
...I started, you know, understanding the connections a little bit better.
2:16
So, Mr Keshe, I want to give you here the floor and.....
2:22
and, introduce yourself maybe. to all those new listeners. Thank you.
2:30
(MK): Thank you very much Keyvan, and welcome to all our children, especially myself.
2:37
I'm still very old but very young at heart, and a child.
2:43
My son is sitting in front of me and he's shaking his head. I was actually about his age, about eight,
2:50
when I was introduced to the X-rays and radiation, and then, I watched the man, at very young age,
2:57
landing on the Moon, and I remember that night, lying on the floor,
3:02
and watching, hoping that I can see the man landing on the Moon. I still clearly remember that day; it was in July, summer time.
3:10
And that was the first time when, to me, was to understand more.
3:17
And he had that, that understanding more about the field and the world, and the galaxies, has never stopped.
3:26
One of the most important things, I think, for the young children to understand, or young listeners,
3:32
is that man is here not just because of what we say.
3:38
Man is here because, collectively, as a race, over hundreds and hundreds and thousands of years,
3:45
we have gathered information and we have passed it from father to son, and to grandchildren.
3:51
And now, we have the facility of the computers to be able to share this knowledge much rapidly and better.
3:59
A lot of you know and have played with magnets. Magnets and the magnetic fields are the backbone of the creation.
4:09
Nothing in the world is, and can exist, without magnetic fields.
"""


def test_detects_workshop_transcript_format() -> None:
    assert looks_like_workshop_transcript(SAMPLE)
    assert not looks_like_workshop_transcript("One paragraph.\n\nAnother paragraph.")


def test_strips_timestamps_and_header() -> None:
    turns = parse_transcript_turns(SAMPLE)
    joined = " ".join(t.text for t in turns)
    assert "0:09" not in joined
    assert "Transkript" not in joined


def test_groups_continuations_and_speakers() -> None:
    turns = parse_transcript_turns(SAMPLE)
    speakers = {t.speaker for t in turns if t.speaker}
    assert {"RC", "KD", "MK"} <= speakers
    mk_turn = next(t for t in turns if t.speaker == "MK" and "magnetic fields" in t.text)
    assert "backbone of the creation" in mk_turn.text


def test_spans_point_into_raw_document() -> None:
    store = DocumentStore()
    doc = store.ingest_text("workshop1.txt", SAMPLE, context="kids")
    segments = clean_transcript_document(doc)
    assert segments
    for seg in segments:
        raw = doc.text[seg.span.start : seg.span.end]
        assert raw.strip()


def test_logistics_dropped_substance_kept() -> None:
    store = DocumentStore()
    doc = store.ingest_text("workshop1.txt", SAMPLE, context="kids")
    cleaned = render_cleaned_transcript(doc, clean_transcript_document(doc))
    assert "Welcome everyone" not in cleaned or "magnetic fields" in cleaned
    assert "magnetic fields are the backbone of the creation" in cleaned.replace("you know, ", "")


def test_mk_export_keeps_full_teaching() -> None:
    store = DocumentStore()
    doc = store.ingest_text("workshop1.txt", SAMPLE, context="kids")
    mk = render_mk_transcript(doc)
    assert "MK:" not in mk
    assert "magnetic fields are the backbone of the creation" in mk
    assert "Nothing in the world is" in mk


WORKSHOP2_SNIPPET = """Rick Crammond (RC) Okay! Welcome everyone to the 2nd Kids Knowledge Seekers Workshop
0:13
from the Spaceship Institute in Italy
0:19
and brought to you by the Keshe Foundation. And, we'll be speaking with Mr Keshe (MK) of the Keshe Foundation
1:06
Okay! Keyvan, are you there ? Would you like to say something now ? (KD) Keyvan Davani again... Mr Keshe...
1:14
As we learnt last time in the 1st Kid's Workshop about what is
2:27
Thank you. (MK) Thank you very much. What we try to do, instead of going into more complicated situations,
2:38
is trying to explain, and go back to it in a few minutes, just for a few minutes,
2:53
What we said is that...the... The world is made of gravitational magnetic fields
3:02
and the whole Universe is built on the interaction of the fields.
"""


def test_workshop2_format_parses_mk_without_colon() -> None:
    turns = parse_transcript_turns(WORKSHOP2_SNIPPET)
    speakers = {t.speaker for t in turns if t.speaker}
    assert "MK" in speakers
    assert "KD" in speakers
    mk = next(t for t in turns if t.speaker == "MK")
    assert "gravitational magnetic fields" in mk.text
    assert "Mr Keshe (MK) of the" not in mk.text


def test_workshop2_mk_export() -> None:
    store = DocumentStore()
    doc = store.ingest_text("workshop2.txt", WORKSHOP2_SNIPPET, context="kids")
    mk = render_mk_transcript(doc)
    assert "gravitational magnetic fields" in mk
    assert len(mk) > 200


def test_kids_corpus_has_22_files() -> None:
    files = sorted(KIDS_DIR.glob("*.txt"))
    assert len(files) == 22


@pytest.mark.slow
def test_full_corpus_distills_deterministically() -> None:
    if not KIDS_DIR.is_dir():
        return
    store_a = DocumentStore()
    store_b = DocumentStore()
    store_a.ingest_directory(KIDS_DIR, context="kids-plasma")
    store_b.ingest_directory(KIDS_DIR, context="kids-plasma")
    result_a = KDPipeline().distill(store_a)
    result_b = KDPipeline().distill(store_b)
    assert result_a == result_b
    assert result_a.documents == 22
    assert result_a.segments > 100
    assert len(result_a.units) > 50
