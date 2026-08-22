"""Tests for the ingestion-status startup resume + GC sweep (todo 4).

Covers the ``after_lizard_bootstrap`` hook and the per-agent background pass:
- stale ``uploaded``/``processing`` entries are re-ingested (status -> completed);
- fresh entries are left untouched;
- deleted-source entries are purged by the GC sweep;
- Redis-down is logged and skipped, never crashing bootstrap.
"""
import asyncio
import time
from unittest.mock import AsyncMock

from cat.core_plugins.ingestion_status import plugin as ingestion_plugin
from cat.core_plugins.ingestion_status import resume
from cat.core_plugins.ingestion_status.registry import (
    IngestionStatus,
    get_status,
    set_status,
    status_key,
)
from cat.core_plugins.ingestion_status.reconcile import reconcile_agent
from cat.db import crud
from tests.utils import agent_id


def _seed_status(agent_key: str, source: str, status: str, updated_at: float) -> None:
    """Seed a status doc with an explicit ``updated_at`` (bypasses set_status)."""
    import asyncio as _asyncio

    async def _store():
        await crud.store(status_key(agent_key, "agent", source), {
            "source": source,
            "scope": "agent",
            "chat_id": None,
            "type": "file",
            "status": status,
            "error": None,
            "error_at": None,
            "created_at": updated_at,
            "updated_at": updated_at,
        })

    _asyncio.get_event_loop().run_until_complete(_store())


async def test_resume_stale_processing(cheshire_cat, monkeypatch):
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key

    # seed a stale processing status (updated_at far in the past)
    old = time.time() - 1000
    await crud.store(status_key(agent_key, "agent", "stale.pdf"), {
        "source": "stale.pdf",
        "scope": "agent",
        "chat_id": None,
        "type": "file",
        "status": "processing",
        "error": None,
        "error_at": None,
        "created_at": old,
        "updated_at": old,
    })

    # resolve the agent to the fixture's ccat and stub the file read
    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    monkeypatch.setattr(cheshire_cat.file_manager, "read_file", lambda source, remote: b"stale content")

    # mock ingest_file to record the call and simulate the completion hook
    calls = []

    async def fake_ingest_file(cat, file, metadata, filename=None, **kwargs):
        calls.append((cat, file, filename))
        await ingestion_plugin.after_rabbithole_stored_documents.function(filename, [object()], cat)

    monkeypatch.setattr(lizard.rabbit_hole, "ingest_file", fake_ingest_file)

    await resume._resume_agent(lizard, agent_key)

    assert len(calls) == 1
    assert calls[0][2] == "stale.pdf"
    doc = await get_status(agent_key, "agent", "stale.pdf")
    assert doc is not None
    assert doc["status"] == "completed"


async def test_resume_leaves_fresh_alone(cheshire_cat, monkeypatch):
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key

    # seed a fresh processing entry (updated_at now)
    now = time.time()
    await crud.store(status_key(agent_key, "agent", "fresh.pdf"), {
        "source": "fresh.pdf",
        "scope": "agent",
        "chat_id": None,
        "type": "file",
        "status": "processing",
        "error": None,
        "error_at": None,
        "created_at": now,
        "updated_at": now,
    })

    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    calls = []

    async def fake_ingest_file(cat, file, metadata, filename=None, **kwargs):
        calls.append((cat, file, filename))

    monkeypatch.setattr(lizard.rabbit_hole, "ingest_file", fake_ingest_file)

    await resume._resume_agent(lizard, agent_key)

    assert calls == []
    doc = await get_status(agent_key, "agent", "fresh.pdf")
    assert doc is not None
    assert doc["status"] == "processing"


async def test_resume_skips_completed(cheshire_cat, monkeypatch):
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key

    # a completed entry, even if stale, must never be re-ingested
    old = time.time() - 1000
    await crud.store(status_key(agent_key, "agent", "done.pdf"), {
        "source": "done.pdf",
        "scope": "agent",
        "chat_id": None,
        "type": "file",
        "status": "completed",
        "error": None,
        "error_at": None,
        "created_at": old,
        "updated_at": old,
    })

    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    calls = []

    async def fake_ingest_file(cat, file, metadata, filename=None, **kwargs):
        calls.append((cat, file, filename))

    monkeypatch.setattr(lizard.rabbit_hole, "ingest_file", fake_ingest_file)

    await resume._resume_agent(lizard, agent_key)

    assert calls == []


async def test_gc_purges_deleted_source(cheshire_cat, monkeypatch):
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key

    await set_status(agent_key, "agent", "deleted.pdf", type_="file", status=IngestionStatus.COMPLETED)

    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    # the file manager lists no files -> the source is absent -> purged
    monkeypatch.setattr(cheshire_cat.file_manager, "list_files", lambda path: [])

    purged = await reconcile_agent(agent_key)

    assert len(purged) == 1
    assert purged[0]["source"] == "deleted.pdf"
    assert await get_status(agent_key, "agent", "deleted.pdf") is None


async def test_gc_keeps_existing_source(cheshire_cat, monkeypatch):
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key

    await set_status(agent_key, "agent", "present.pdf", type_="file", status=IngestionStatus.COMPLETED)

    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    # the file manager still lists the source -> kept
    monkeypatch.setattr(cheshire_cat.file_manager, "list_files", lambda path: [type("F", (), {"name": "present.pdf"})()])

    purged = await reconcile_agent(agent_key)

    assert purged == []
    assert await get_status(agent_key, "agent", "present.pdf") is not None


async def test_gc_keeps_error_entry_for_absent_source(cheshire_cat, monkeypatch):
    """M3: an ``error`` entry for an absent source survives the reconcile.

    A failed upload never lands in the file manager, so without the carve-out
    it would be purged on first read and the error badge could never appear.
    A ``completed`` entry for an absent source is still purged.
    """
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key

    await set_status(agent_key, "agent", "failed.pdf", type_="file", status=IngestionStatus.ERROR, error="boom")
    await set_status(agent_key, "agent", "deleted.pdf", type_="file", status=IngestionStatus.COMPLETED)

    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    # the file manager lists no files -> both sources are absent from canonical
    monkeypatch.setattr(cheshire_cat.file_manager, "list_files", lambda path: [])

    purged = await reconcile_agent(agent_key)

    # only the completed entry is purged; the error entry survives
    assert [p["source"] for p in purged] == ["deleted.pdf"]
    assert await get_status(agent_key, "agent", "deleted.pdf") is None
    assert await get_status(agent_key, "agent", "failed.pdf") is not None


async def test_gc_keeps_chat_error_entry_when_conversation_gone(cheshire_cat, monkeypatch):
    """M3: a chat-scoped ``error`` entry survives even when the conversation is gone."""
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key

    await set_status(
        agent_key, "gone_chat", "chat_failed.pdf",
        type_="file", status=IngestionStatus.ERROR, error="boom", chat_id="gone_chat",
    )

    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    monkeypatch.setattr(cheshire_cat.file_manager, "list_files", lambda path: [])

    purged = await reconcile_agent(agent_key, chat_id="gone_chat")

    assert purged == []
    assert await get_status(agent_key, "gone_chat", "chat_failed.pdf") is not None


async def test_redis_down_skips_pass(monkeypatch):
    """Redis unreachable -> the startup pass is skipped without crashing."""

    def boom():
        raise ConnectionError("redis down")

    monkeypatch.setattr("cat.db.cruds.settings.get_async_db", boom)

    # must not raise
    await resume._startup_pass(None)


async def test_after_lizard_bootstrap_schedules_pass(monkeypatch):
    """The hook schedules the pass fire-and-forget (never blocks bootstrap)."""
    scheduled = []

    def fake_ensure_future(coro):
        scheduled.append(coro)

    monkeypatch.setattr(asyncio, "ensure_future", fake_ensure_future)

    await resume.after_lizard_bootstrap.function(None)

    assert len(scheduled) == 1
    # await the scheduled coroutine so it is not left dangling
    await scheduled[0]