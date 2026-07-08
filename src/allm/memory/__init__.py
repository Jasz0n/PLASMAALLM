"""Lifelong Memory: an append-only episodic record of what happened.

Successes, failures, revisions, reasoning traces and observations as a
queryable timeline (confidence history and belief revisions live,
versioned, in the teacher state and knowledge graph). Lexical search
now; vector recall (FAISS/Chroma) is a planned second backend.
"""

from allm.memory.episodic import EpisodicMemory, memory_backends
from allm.memory.types import Episode, EpisodeKind

__all__ = ["EpisodicMemory", "memory_backends", "Episode", "EpisodeKind"]
