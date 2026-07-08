"""Build Knowledge Packages from KDP output or labelled samples."""

from __future__ import annotations

from pathlib import Path

from allm.data.base import Sample
from allm.kdp.corpus import load_samples_jsonl
from allm.kdp.types import ConflictNode, KnowledgeUnit, KnowledgeUnitType
from allm.researcher.types import KnowledgePackage, PackageConcept, PackageConflict


def package_from_distillation(
    result,
    *,
    provider: str,
    title: str,
    provenance: str,
    curriculum_topic: str | None = None,
) -> KnowledgePackage:
    """Convert a KDP distillation run into one Knowledge Package."""
    concepts = _concepts_from_units(result.units)
    definitions = _definitions_from_units(result.units)
    procedures = tuple(u.content for u in result.units if u.type == "procedure")
    examples = tuple(u.content for u in result.units if u.type == "fact")[:32]
    misconceptions = tuple(u.content for u in result.units if u.type == "misconception")
    conflicts = _conflicts_from_nodes(result.conflicts)
    evidence = tuple(sorted({src for u in result.units for src in u.sources}))[:64]
    confidence = _mean_confidence(result.units)

    return KnowledgePackage.build(
        provider=provider,
        title=title,
        concepts=concepts,
        definitions=definitions,
        procedures=procedures,
        examples=examples,
        misconceptions=misconceptions,
        evidence=evidence,
        confidence=confidence,
        provenance=provenance,
        source_refs=tuple(sorted({u.id for u in result.units}))[:64],
        conflicts=conflicts,
        curriculum_topic=curriculum_topic,
    )


def package_from_workshop_dir(
    corpus_dir: Path | str,
    *,
    provider: str = "kids-workshops",
    max_files: int | None = None,
    curriculum_topic: str = "kids-plasma",
) -> KnowledgePackage:
    """Run KDP on workshop transcripts and return one package."""
    from allm.kdp.ingestion import DocumentStore
    from allm.kdp.pipeline import KDPipeline

    directory = Path(corpus_dir)
    paths = sorted(directory.glob("*.txt"))
    if max_files is not None:
        paths = paths[:max_files]
    store = DocumentStore()
    for path in paths:
        store.ingest_file(path, context="kids-plasma")
    result = KDPipeline().distill(store)
    return package_from_distillation(
        result,
        provider=provider,
        title=f"Kids plasma workshops ({len(paths)} files)",
        provenance=str(directory),
        curriculum_topic=curriculum_topic,
    )


def package_from_book_dir(
    books_dir: Path | str,
    *,
    provider: str = "keshe-books",
    max_files: int | None = 1,
    max_pages: int = 32,
    curriculum_topic: str = "kids-plasma",
    pdf_backend: str = "auto",
) -> KnowledgePackage:
    """Run KDP on extracted PDF book text and return one package."""
    directory = Path(books_dir)
    paths = sorted(directory.glob("*.pdf"))
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(directory)
    store = _ingest_book_paths(
        paths,
        curriculum_topic=curriculum_topic,
        max_pages=max_pages,
        pdf_backend=pdf_backend,
    )
    from allm.kdp.pipeline import KDPipeline

    result = KDPipeline().distill(store)
    return package_from_distillation(
        result,
        provider=provider,
        title=f"Keshe books ({len(paths)} pdf)",
        provenance=str(directory),
        curriculum_topic=curriculum_topic,
    )


def package_from_single_book_pdf(
    path: Path | str,
    *,
    provider: str = "keshe-books",
    max_pages: int = 32,
    curriculum_topic: str = "kids-plasma",
    pdf_backend: str = "auto",
) -> KnowledgePackage:
    """Distill one PDF into a Knowledge Package."""
    book_path = Path(path)
    store = _ingest_book_paths(
        [book_path],
        curriculum_topic=curriculum_topic,
        max_pages=max_pages,
        pdf_backend=pdf_backend,
    )
    from allm.kdp.pipeline import KDPipeline

    result = KDPipeline().distill(store)
    return package_from_distillation(
        result,
        provider=provider,
        title=book_path.name,
        provenance=str(book_path),
        curriculum_topic=curriculum_topic,
    )


def _ingest_book_paths(
    paths: list[Path],
    *,
    curriculum_topic: str,
    max_pages: int,
    pdf_backend: str,
):
    from allm.kdp.ingestion import DocumentStore
    from allm.researcher.book_pdf import extract_pdf_text

    store = DocumentStore()
    for path in paths:
        text = extract_pdf_text(path, max_pages=max_pages, backend=pdf_backend)
        if not text.strip():
            continue
        store.ingest_text(path.name, text, context=curriculum_topic)
    return store


def module_docstring(path: Path) -> str | None:
    """First module-level docstring of a Python file, or None."""
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None
    return ast.get_docstring(tree)


def package_from_repository(
    repo_dir: Path | str,
    *,
    provider: str,
    max_files: int | None = 48,
    curriculum_topic: str | None = None,
) -> KnowledgePackage:
    """Distill a real repository into one Knowledge Package (M49).

    Markdown documentation is ingested verbatim; Python sources
    contribute their module docstrings (design intent, written by the
    developers themselves). Everything runs through KDP, so provenance,
    dedup and contradiction detection work exactly as for any corpus.
    """
    from allm.kdp.ingestion import DocumentStore
    from allm.kdp.pipeline import KDPipeline
    from allm.researcher.providers import RepositoryProvider

    directory = Path(repo_dir)
    repo_provider = RepositoryProvider(directory, max_files=max_files)
    paths = repo_provider.discover()
    if not paths:
        raise FileNotFoundError(f"no knowledge-bearing files under {directory}")
    context = f"repository:{directory.name}"
    store = DocumentStore()
    documents = 0
    for path in paths:
        rel = path.relative_to(directory)
        if path.suffix == ".py":
            doc = module_docstring(path)
            if doc:
                store.ingest_text(str(rel), f"Module {rel}: {doc}", context=context)
                documents += 1
        else:
            store.ingest_text(str(rel), path.read_text(encoding="utf-8"), context=context)
            documents += 1
    result = KDPipeline().distill(store)
    return package_from_distillation(
        result,
        provider=provider,
        title=f"Repository {directory.name} ({documents} documents)",
        provenance=str(directory),
        curriculum_topic=curriculum_topic or f"software-{directory.name.lower()}",
    )


def package_from_samples_jsonl(
    path: Path | str,
    *,
    provider: str,
    title: str,
) -> KnowledgePackage:
    """Build a package from a labelled sample pool (software fixture)."""
    samples = load_samples_jsonl(path)
    by_topic: dict[str, list[Sample]] = {}
    for sample in samples:
        topic = str(sample.metadata.get("topic", "general"))
        by_topic.setdefault(topic, []).append(sample)

    concepts = tuple(
        PackageConcept(name=topic, description=rows[0].target or "", confidence=0.7)
        for topic, rows in sorted(by_topic.items())
        if rows
    )
    definitions = tuple(
        (row.input, row.target or "")
        for row in samples
        if row.target and row.metadata.get("sample_kind") == "definition"
    )
    teaching = tuple(
        row.target or row.input
        for row in samples
        if row.metadata.get("sample_kind") == "teaching"
    )

    return KnowledgePackage.build(
        provider=provider,
        title=title,
        concepts=concepts,
        definitions=definitions,
        examples=teaching[:16],
        confidence=0.75,
        provenance=str(path),
        source_refs=tuple(row.id for row in samples),
    )


def _concepts_from_units(units: tuple[KnowledgeUnit, ...]) -> tuple[PackageConcept, ...]:
    seen: dict[str, PackageConcept] = {}
    for unit in units:
        if unit.type != "concept":
            continue
        name = unit.normalized_concept or unit.content[:48]
        if name not in seen:
            seen[name] = PackageConcept(
                name=name,
                description=unit.content,
                confidence=unit.confidence,
            )
    return tuple(seen.values())


def _definitions_from_units(units: tuple[KnowledgeUnit, ...]) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for unit in units:
        if unit.type == "concept" and unit.content:
            term = unit.normalized_concept or "concept"
            rows.append((term, unit.content))
    return tuple(rows[:64])


def _conflicts_from_nodes(nodes: tuple[ConflictNode, ...]) -> tuple[PackageConflict, ...]:
    return tuple(
        PackageConflict(
            concept=node.concept,
            status="disagreement",
            sources=node.sources,
            detail=f"{node.interpretation_a} vs {node.interpretation_b}",
        )
        for node in nodes
    )


def _mean_confidence(units: tuple[KnowledgeUnit, ...]) -> float:
    if not units:
        return 0.5
    return round(sum(u.confidence for u in units) / len(units), 4)
