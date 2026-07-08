"""Tests for allm.memory: append, recall filters, search, exam bridge."""

from pathlib import Path

import pytest

from allm.exam import Answer, ExactMatchGrader, ExamResult, Question
from allm.memory import Episode, EpisodicMemory, memory_backends
from allm.storage import SQLiteRecordStore


@pytest.fixture()
def memory(tmp_path: Path):
    store = SQLiteRecordStore(tmp_path / "memory.sqlite3")
    yield EpisodicMemory(store)
    store.close()


def test_remember_and_recall_filters(memory: EpisodicMemory) -> None:
    memory.remember("s1", "success", "solved integral", topic="math")
    memory.remember("s1", "failure", "confused derivative", topic="math")
    memory.remember("s2", "failure", "wrong capital", topic="geography")

    assert len(memory.recall()) == 3
    assert [e.summary for e in memory.recall(actor="s1", kind="failure")] == [
        "confused derivative"
    ]
    assert [e.actor for e in memory.recall(topic="geography")] == ["s2"]
    assert len(memory.recall(actor="s1", limit=1)) == 1


def test_episode_ids_are_unique_and_ordered(memory: EpisodicMemory) -> None:
    first = memory.remember("s1", "observation", "a")
    second = memory.remember("s1", "observation", "b")
    assert first.id != second.id
    assert [e.id for e in memory.recall()] == [first.id, second.id]


def test_sequence_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    store = SQLiteRecordStore(path)
    EpisodicMemory(store).remember("s1", "observation", "before restart")
    store.close()
    store = SQLiteRecordStore(path)
    revived = EpisodicMemory(store)
    episode = revived.remember("s1", "observation", "after restart")
    assert len(revived.recall()) == 2  # no id collision swallowed an episode
    assert episode.id not in [e.id for e in revived.recall()[:-1]]
    store.close()


def test_search_ranks_by_overlap(memory: EpisodicMemory) -> None:
    memory.remember("s1", "failure", "confused gravity with magnetism")
    memory.remember("s1", "success", "explained gravity correctly")
    memory.remember("s1", "observation", "read about chemistry")

    hits = memory.search("gravity confused")
    assert hits[0].summary == "confused gravity with magnetism"
    assert len(hits) == 2  # chemistry episode does not match
    assert memory.search("") == []


def test_remember_exam_creates_success_and_failure(memory: EpisodicMemory) -> None:
    grader = ExactMatchGrader()
    q1 = Question(id="q1", prompt="2+2?", expected="4", topic="math")
    q2 = Question(id="q2", prompt="Capital of France?", expected="Paris", topic="geo")
    result = ExamResult(
        exam_id="e1",
        student_id="s1",
        results=(
            grader.grade(q1, Answer(question_id="q1", text="4", confidence=0.9,
                                    reasoning="basic arithmetic")),
            grader.grade(q2, Answer(question_id="q2", text="Lyon", confidence=0.6)),
        ),
    )
    episodes = memory.remember_exam(result)
    assert [e.kind for e in episodes] == ["success", "failure"]
    assert episodes[0].detail["reasoning"] == "basic arithmetic"
    assert episodes[1].detail["expected"] == "Paris"
    assert episodes[1].confidence == 0.6


def test_appending_same_id_versions_not_overwrites(
    memory: EpisodicMemory, tmp_path: Path
) -> None:
    episode = Episode(id="fixed", actor="s1", kind="observation", summary="v1")
    memory.append(episode)
    memory.append(episode.model_copy(update={"summary": "v2"}))
    recalled = [e for e in memory.recall() if e.id == "fixed"]
    assert len(recalled) == 1 and recalled[0].summary == "v2"  # latest wins on read


def test_backend_registered() -> None:
    assert memory_backends.get("episodic") is EpisodicMemory
