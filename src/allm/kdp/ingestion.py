"""Stage 1 — Ingestion.

Documents get content-derived ids and are optionally persisted raw into
the record store (namespace ``documents``), so every later span
reference can always be resolved against the exact original bytes.
Raw text is never modified — cleaning happens on copies downstream.
"""

from __future__ import annotations

from pathlib import Path

from allm.core.logging import get_logger
from allm.kdp.types import Document, content_hash
from allm.storage.base import RecordStore

logger = get_logger("kdp.ingestion")


class DocumentStore:
    """Holds ingested documents; optionally persists them versioned."""

    def __init__(self, store: RecordStore | None = None) -> None:
        self._documents: dict[str, Document] = {}
        self._store = store

    def ingest_text(self, name: str, text: str, *, context: str = "general") -> Document:
        """Ingest one raw text stream under a stable, content-derived id."""
        doc_id = f"doc_{content_hash(name, text)}"
        if doc_id in self._documents:
            return self._documents[doc_id]  # identical name+text: same doc
        document = Document(id=doc_id, name=name, text=text, context=context)
        self._documents[doc_id] = document
        if self._store is not None:
            self._store.put(
                "documents",
                doc_id,
                {"name": name, "text": text, "context": context},
                reason="kdp ingestion (raw, never modified)",
            )
        logger.info("ingested %s as %s (%d chars)", name, doc_id, len(text))
        return document

    def ingest_file(self, path: Path | str, *, context: str | None = None) -> Document:
        """Ingest a text/markdown file; context defaults to the file stem."""
        path = Path(path)
        return self.ingest_text(
            path.name,
            path.read_text(encoding="utf-8"),
            context=context if context is not None else path.stem,
        )

    def ingest_directory(
        self,
        directory: Path | str,
        *,
        pattern: str = "*.txt",
        context: str | None = None,
    ) -> list[Document]:
        """Ingest all matching files under ``directory``, sorted by name."""
        root = Path(directory)
        docs: list[Document] = []
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                docs.append(
                    self.ingest_file(
                        path,
                        context=context if context is not None else root.name,
                    )
                )
        logger.info("ingested %d file(s) from %s", len(docs), root)
        return docs

    def get(self, doc_id: str) -> Document | None:
        return self._documents.get(doc_id)

    def documents(self) -> list[Document]:
        """All documents, ordered by id for determinism."""
        return [self._documents[k] for k in sorted(self._documents)]
