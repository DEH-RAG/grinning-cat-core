"""Tests for the chunk-reuse re-embed path in ``CheshireCat.embed_stored_sources``.

The embedder-change re-embed (``embed_all_in_cheshire_cats``) used to re-parse every
source from disk/URL via ``rabbit_hole.ingest_file``. This suite pins the new
chunk-reuse flow: existing stored chunks are reused and only the vectors are
recomputed, through the ``BaseVectorDatabaseHandler`` interface only (no Qdrant
classes). A fake handler implementing the interface stands in for the vector DB.
"""
import hashlib

from langchain_core.documents import Document

from cat.db import crud
from cat.looking_glass.models import StoredSourceWithMetadata
from cat.rabbit_hole import RabbitHole
from cat.services.memory.models import Record, VectorMemoryType

from tests.utils import agent_id


class FakeEmbedder:
    """Minimal embedder exposing the attributes the reuse path reads."""

    name = "FakeEmbedder"
    size = 4
    max_input_tokens = 1000

    def embed_documents(self, texts):
        return [[0.1] * self.size for _ in texts]


class FailingEmbedder(FakeEmbedder):
    """Embedder that raises when asked to embed a specific text."""

    def __init__(self, boom_text):
        self.boom_text = boom_text

    def embed_documents(self, texts):
        for t in texts:
            if self.boom_text in t:
                raise RuntimeError(f"cannot embed {self.boom_text!r}")
        return super().embed_documents(texts)


class FakeVectorHandler:
    """In-memory ``BaseVectorDatabaseHandler`` interface implementation.

    Only the interface methods used by the chunk-reuse path are implemented.
    """

    def __init__(self, points=None):
        self.points = list(points or [])
        self.added = []
        self.deleted = []

    async def get_all_tenant_points(
        self, collection_name, limit=None, offset=None, metadata=None, with_vectors=True
    ):
        return list(self.points), None

    async def add_points_to_tenant(self, collection_name, points):
        self.added.append((collection_name, points))

    async def delete_tenant_points(self, collection_name, metadata=None):
        self.deleted.append((collection_name, metadata))


def _record(source, page_content, when=123.0, file_hash="abc", chat_id=None):
    """Build a stored ``Record`` with the payload shape a normal ingest produces."""
    metadata = {"source": source, "when": when, "hash": file_hash}
    if chat_id:
        metadata["chat_id"] = chat_id
    return Record(
        id=hashlib.sha256(f"{source}:{page_content}".encode()).hexdigest(),
        payload={
            "id": "some-id",
            "page_content": page_content,
            "metadata": metadata,
            "tenant_id": agent_id,
        },
        vector=[0.1, 0.2, 0.3, 0.4],
    )


def _source(name, chat_id=None):
    metadata = {"source": name}
    if chat_id:
        metadata["chat_id"] = chat_id
    return StoredSourceWithMetadata(name=name, path=name, content=None, metadata=metadata)


def _status_key(source, scope="agent"):
    digest = hashlib.sha256(source.encode()).hexdigest()
    return f"agents:{agent_id}:ingestion:{scope}:{digest}"


async def _install_embedder(cheshire_cat, monkeypatch, handler, embedder):
    """Wire the fake handler + embedder onto the cat and return the ingest spy."""
    monkeypatch.setattr(cheshire_cat, "vector_memory_handler", handler)

    async def fake_embedder():
        return embedder

    monkeypatch.setattr(cheshire_cat, "embedder", fake_embedder)

    ingest_calls = []

    async def spy_ingest_file(cat, file, filename, metadata=None, store_file=False, content_type=None):
        ingest_calls.append(filename)

    monkeypatch.setattr(cheshire_cat.rabbit_hole, "ingest_file", spy_ingest_file)
    return ingest_calls


async def test_reuse_does_not_call_ingest_file(cheshire_cat, monkeypatch):
    """(a) With points present, ingest_file is NOT called; chunks are re-embedded
    and re-stored with the same payload shape."""
    fake = FakeVectorHandler(points=[
        _record("test.txt", "chunk one"),
        _record("test.txt", "chunk two"),
    ])
    ingest_calls = await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())

    await cheshire_cat.embed_stored_sources(
        VectorMemoryType.DECLARATIVE, [_source("test.txt")],
    )

    # no re-parse from disk/URL
    assert ingest_calls == []

    # exactly one add_points_to_tenant call with the two re-embedded chunks
    assert len(fake.added) == 1
    collection, points = fake.added[0]
    assert collection == str(VectorMemoryType.DECLARATIVE)
    assert len(points) == 2

    # payload shape matches a normal ingest: page_content + metadata{source,when,hash}
    for point in points:
        assert point.payload["page_content"] in ("chunk one", "chunk two")
        meta = point.payload["metadata"]
        assert meta["source"] == "test.txt"
        assert "when" in meta
        assert "hash" in meta
        # vectors were recomputed with the new embedder's dimension
        assert len(point.vector) == FakeEmbedder.size

    # the source's old points are deleted (metadata-filtered) before the
    # re-embedded points are added, so they replace rather than duplicate
    assert fake.deleted == [(str(VectorMemoryType.DECLARATIVE), {"source": "test.txt"})]

    # the delete happens BEFORE the add, so the re-embedded points replace the old ones
    assert fake.deleted and fake.added


async def test_reuse_invokes_split_oversized(cheshire_cat, monkeypatch):
    """(b) _split_oversized is invoked when the new embedder limit < chunk size."""
    big_chunk = "x" * 5000
    fake = FakeVectorHandler(points=[_record("big.txt", big_chunk)])
    await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())

    calls = []

    def recording_split(self, docs, embedder):
        calls.append((list(docs), embedder))
        return docs

    monkeypatch.setattr(RabbitHole, "_split_oversized", recording_split)

    await cheshire_cat.embed_stored_sources(
        VectorMemoryType.DECLARATIVE, [_source("big.txt")],
    )

    assert len(calls) == 1
    docs, embedder = calls[0]
    assert len(docs) == 1
    assert docs[0].page_content == big_chunk
    assert embedder is not None


async def test_empty_lookup_falls_back_to_ingest_file(cheshire_cat, monkeypatch):
    """(c) Empty lookups fall back to ingest_file."""
    fake = FakeVectorHandler(points=[])  # no reusable chunks
    ingest_calls = await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())

    await cheshire_cat.embed_stored_sources(
        VectorMemoryType.DECLARATIVE, [_source("test.txt")],
    )

    assert ingest_calls == ["test.txt"]
    # no chunk-reuse add happened
    assert fake.added == []


async def test_status_processing_then_completed(cheshire_cat, monkeypatch):
    """(d) Per-source status is written processing -> completed."""
    fake = FakeVectorHandler(points=[_record("test.txt", "chunk one")])
    await _install_embedder(cheshire_cat, monkeypatch, fake, FakeEmbedder())

    writes = []

    async def recording_store(key, value, path="$", nx=False, xx=False, expire=None):
        writes.append((key, value))
        return value

    monkeypatch.setattr(crud, "store", recording_store)

    await cheshire_cat.embed_stored_sources(
        VectorMemoryType.DECLARATIVE, [_source("test.txt")],
    )

    key = _status_key("test.txt")
    statuses = [v["status"] for k, v in writes if k == key]
    assert statuses == ["processing", "completed"]

    # the completed payload carries the full schema
    completed = [v for k, v in writes if k == key and v["status"] == "completed"][0]
    assert completed["source"] == "test.txt"
    assert completed["scope"] == "agent"
    assert completed["chat_id"] is None
    assert completed["type"] == "file"
    assert completed["error"] is None
    assert "created_at" in completed
    assert "updated_at" in completed


async def test_one_source_failing_marks_error_others_complete(cheshire_cat, monkeypatch):
    """(e) A failing source is marked error; other sources still complete."""
    fake = FakeVectorHandler(points=[
        _record("good.txt", "good chunk"),
        _record("boom.txt", "boom chunk"),
    ])
    await _install_embedder(cheshire_cat, monkeypatch, fake, FailingEmbedder("boom chunk"))

    writes = []

    async def fake_store(key, value, path=None, nx=False, xx=False, expire=None):
        writes.append((key, value))
        return value

    monkeypatch.setattr(crud, "store", fake_store)

    await cheshire_cat.embed_stored_sources(
        VectorMemoryType.DECLARATIVE, [_source("good.txt"), _source("boom.txt")],
    )

    good_statuses = [v["status"] for k, v in writes if k == _status_key("good.txt")]
    boom_statuses = [v["status"] for k, v in writes if k == _status_key("boom.txt")]

    assert good_statuses == ["processing", "completed"]
    assert boom_statuses == ["processing", "error"]

    boom_error = [v for k, v in writes if k == _status_key("boom.txt") and v["status"] == "error"][0]
    assert "cannot embed" in boom_error["error"]
    assert boom_error["error_at"] is not None