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

import pytest

from cat.core_plugins.ingestion_status import plugin as ingestion_plugin
from cat.core_plugins.ingestion_status import resume
from cat.core_plugins.ingestion_status.reconcile import reconcile_agent
from cat.core_plugins.ingestion_status.registry import (
    IngestionStatus,
    get_status,
    set_status,
    status_key,
)
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
    # the file is present on disk (a fresh entry is being actively processed)
    monkeypatch.setattr(cheshire_cat.file_manager, "read_file", lambda source, remote: b"content")
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
    """The hook schedules the startup pass AND the periodic sweep fire-and-forget."""
    scheduled = []

    def fake_ensure_future(coro):
        scheduled.append(coro)

    monkeypatch.setattr(asyncio, "ensure_future", fake_ensure_future)
    monkeypatch.setenv("CAT_INGESTION_RESUME_INTERVAL_SECONDS", "60")
    # keep the startup pass a no-op (no agents to sweep in this test)
    monkeypatch.setattr(
        "cat.core_plugins.ingestion_status.resume.crud_settings.get_agents_main_keys",
        AsyncMock(return_value=[]),
    )

    await resume.after_lizard_bootstrap.function(None)

    assert len(scheduled) == 2
    # both the immediate startup pass and the periodic sweep are scheduled
    assert [c.cr_code.co_name for c in scheduled] == ["_startup_pass", "_periodic_sweep_loop"]
    # await the startup pass so it is not left dangling
    await scheduled[0]
    # the periodic sweep loops forever: close it instead of awaiting
    scheduled[1].close()


async def test_resume_retries_stale_error_with_file_present(cheshire_cat, monkeypatch):
    """An error entry with the file still on disk is re-ingested when stale."""
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key

    old = time.time() - 1000
    await crud.store(status_key(agent_key, "agent", "errored.pdf"), {
        "source": "errored.pdf",
        "scope": "agent",
        "chat_id": None,
        "type": "file",
        "status": "error",
        "error": "parse failed",
        "error_at": old,
        "created_at": old,
        "updated_at": old,
    })

    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    monkeypatch.setattr(cheshire_cat.file_manager, "read_file", lambda source, remote: b"content")

    calls = []

    async def fake_ingest_file(cat, file, metadata, filename=None, **kwargs):
        calls.append((cat, file, filename))
        await ingestion_plugin.after_rabbithole_stored_documents.function(filename, [object()], cat)

    monkeypatch.setattr(lizard.rabbit_hole, "ingest_file", fake_ingest_file)

    await resume._resume_agent(lizard, agent_key)

    assert len(calls) == 1
    assert calls[0][2] == "errored.pdf"
    doc = await get_status(agent_key, "agent", "errored.pdf")
    assert doc["status"] == "completed"


async def test_resume_skips_error_with_file_missing(cheshire_cat, monkeypatch):
    """An error entry whose file is gone is NOT re-ingested (nothing to read);
    the teacher must re-upload."""
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key

    old = time.time() - 1000
    await crud.store(status_key(agent_key, "agent", "lost.pdf"), {
        "source": "lost.pdf",
        "scope": "agent",
        "chat_id": None,
        "type": "file",
        "status": "error",
        "error": "boom",
        "error_at": old,
        "created_at": old,
        "updated_at": old,
    })

    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    monkeypatch.setattr(cheshire_cat.file_manager, "read_file", lambda source, remote: None)

    calls = []

    async def fake_ingest_file(cat, file, metadata, filename=None, **kwargs):
        calls.append(filename)

    monkeypatch.setattr(lizard.rabbit_hole, "ingest_file", fake_ingest_file)

    await resume._resume_agent(lizard, agent_key)

    assert calls == []
    doc = await get_status(agent_key, "agent", "lost.pdf")
    assert doc["status"] == "error"


async def test_fresh_processing_claimed_by_other_worker_not_double_ingested(cheshire_cat, monkeypatch):
    """Two workers seeing the same fresh processing entry: only the one that
    wins the per-source claim proceeds — the other sees the row claimed
    (owner set + updated_at bumped) and skips."""
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key

    old = time.time() - 1000
    await crud.store(status_key(agent_key, "agent", "same.pdf"), {
        "source": "same.pdf",
        "scope": "agent",
        "chat_id": None,
        "type": "file",
        "status": "processing",
        "error": None,
        "error_at": None,
        "created_at": old,
        "updated_at": old,
    })

    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    monkeypatch.setattr(cheshire_cat.file_manager, "read_file", lambda source, remote: b"content")

    calls = []

    async def fake_ingest_file(cat, file, metadata, filename=None, **kwargs):
        calls.append(filename)
        await ingestion_plugin.after_rabbithole_stored_documents.function(filename, [object()], cat)

    monkeypatch.setattr(lizard.rabbit_hole, "ingest_file", fake_ingest_file)

    # Worker 1 wins the claim, Worker 2 runs right after (row now fresh+claimed)
    await resume._resume_agent(lizard, agent_key)
    await resume._resume_agent(lizard, agent_key)

    assert len(calls) == 1
    doc = await get_status(agent_key, "agent", "same.pdf")
    assert doc["status"] == "completed"


async def test_resume_marks_stale_processing_error_when_file_missing(cheshire_cat, monkeypatch):
    """A stale ``processing`` entry whose file is gone is marked ``error`` (pre-check)."""
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key

    old = time.time() - 1000
    await crud.store(status_key(agent_key, "agent", "gone.pdf"), {
        "source": "gone.pdf",
        "scope": "agent",
        "chat_id": None,
        "type": "file",
        "status": "processing",
        "error": None,
        "error_at": None,
        "created_at": old,
        "updated_at": old,
    })

    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    monkeypatch.setattr(cheshire_cat.file_manager, "read_file", lambda source, remote: None)

    calls = []

    async def fake_ingest_file(cat, file, metadata, filename=None, **kwargs):
        calls.append(filename)

    monkeypatch.setattr(lizard.rabbit_hole, "ingest_file", fake_ingest_file)

    await resume._resume_agent(lizard, agent_key)

    assert calls == []
    doc = await get_status(agent_key, "agent", "gone.pdf")
    assert doc is not None
    assert doc["status"] == "error"
    assert doc["error"] == "Source file does not exist on disk; cannot resume. Remove the file to abandon it."


async def test_resume_marks_stale_uploaded_error_when_file_missing(cheshire_cat, monkeypatch):
    """A stale ``uploaded`` entry whose file is gone is marked ``error`` (pre-check)."""
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key

    old = time.time() - 1000
    await crud.store(status_key(agent_key, "agent", "gone_upload.pdf"), {
        "source": "gone_upload.pdf",
        "scope": "agent",
        "chat_id": None,
        "type": "file",
        "status": "uploaded",
        "error": None,
        "error_at": None,
        "created_at": old,
        "updated_at": old,
    })

    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    monkeypatch.setattr(cheshire_cat.file_manager, "read_file", lambda source, remote: None)

    calls = []

    async def fake_ingest_file(cat, file, metadata, filename=None, **kwargs):
        calls.append(filename)

    monkeypatch.setattr(lizard.rabbit_hole, "ingest_file", fake_ingest_file)

    await resume._resume_agent(lizard, agent_key)

    assert calls == []
    doc = await get_status(agent_key, "agent", "gone_upload.pdf")
    assert doc is not None
    assert doc["status"] == "error"
    assert doc["error"] == "Source file does not exist on disk; cannot resume. Remove the file to abandon it."


async def test_resume_marks_error_when_file_missing_at_read_step(cheshire_cat, monkeypatch):
    """File present at pre-check but gone at the re-ingest read -> marked ``error``."""
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key

    old = time.time() - 1000
    await crud.store(status_key(agent_key, "agent", "vanished.pdf"), {
        "source": "vanished.pdf",
        "scope": "agent",
        "chat_id": None,
        "type": "file",
        "status": "processing",
        "error": None,
        "error_at": None,
        "created_at": old,
        "updated_at": old,
    })

    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    reads = {"count": 0}

    def flaky_read(source, remote):
        reads["count"] += 1
        return b"content" if reads["count"] == 1 else None

    monkeypatch.setattr(cheshire_cat.file_manager, "read_file", flaky_read)

    calls = []

    async def fake_ingest(cat, file, metadata, filename=None, **kwargs):
        calls.append(filename)

    monkeypatch.setattr(lizard.rabbit_hole, "ingest_file", fake_ingest)

    await resume._resume_agent(lizard, agent_key)

    assert calls == []
    doc = await get_status(agent_key, "agent", "vanished.pdf")
    assert doc is not None
    assert doc["status"] == "error"
    assert doc["error"] == "Source file does not exist on disk; cannot resume. Remove the file to abandon it."


async def test_resume_does_not_mark_url_error(cheshire_cat, monkeypatch):
    """A stale URL entry is re-downloaded, never marked ``error`` for a missing file."""
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key

    old = time.time() - 1000
    await crud.store(status_key(agent_key, "agent", "https://example.com/doc.pdf"), {
        "source": "https://example.com/doc.pdf",
        "scope": "agent",
        "chat_id": None,
        "type": "url",
        "status": "processing",
        "error": None,
        "error_at": None,
        "created_at": old,
        "updated_at": old,
    })

    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))

    calls = []

    async def fake_ingest(cat, file, metadata, filename=None, **kwargs):
        calls.append(filename)

    monkeypatch.setattr(lizard.rabbit_hole, "ingest_file", fake_ingest)

    await resume._resume_agent(lizard, agent_key)

    assert calls == ["https://example.com/doc.pdf"]
    doc = await get_status(agent_key, "agent", "https://example.com/doc.pdf")
    assert doc is not None
    assert doc["status"] != "error"


async def test_resume_completes_stale_uploaded_when_file_on_disk(cheshire_cat, monkeypatch):
    """A stale ``uploaded`` entry whose file is on disk is resumed to ``completed``."""
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key

    old = time.time() - 1000
    await crud.store(status_key(agent_key, "agent", "on_disk.pdf"), {
        "source": "on_disk.pdf",
        "scope": "agent",
        "chat_id": None,
        "type": "file",
        "status": "uploaded",
        "error": None,
        "error_at": None,
        "created_at": old,
        "updated_at": old,
    })

    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    # the file is present on disk
    monkeypatch.setattr(cheshire_cat.file_manager, "read_file", lambda source, remote: b"content")

    calls = []

    async def fake_ingest_file(cat, file, metadata, filename=None, **kwargs):
        calls.append((cat, file, filename))
        await ingestion_plugin.after_rabbithole_stored_documents.function(filename, [object()], cat)

    monkeypatch.setattr(lizard.rabbit_hole, "ingest_file", fake_ingest_file)

    await resume._resume_agent(lizard, agent_key)

    assert len(calls) == 1
    assert calls[0][2] == "on_disk.pdf"
    doc = await get_status(agent_key, "agent", "on_disk.pdf")
    assert doc is not None
    assert doc["status"] == "completed"


async def test_periodic_sweep_completes_stale_uploaded(cheshire_cat, monkeypatch):
    """The periodic sweep loop re-runs the startup pass and completes a stale ``uploaded`` entry.

    ``_periodic_sweep_loop`` must recover a stale ``uploaded`` entry without a
    manual restart: one loop iteration runs the startup pass, which re-ingests
    the source and transitions it to ``completed``. The loop is stopped after
    its first pass by making the post-pass ``asyncio.sleep`` raise
    ``CancelledError``.
    """
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key

    old = time.time() - 1000
    await crud.store(status_key(agent_key, "agent", "sweep.pdf"), {
        "source": "sweep.pdf",
        "scope": "agent",
        "chat_id": None,
        "type": "file",
        "status": "uploaded",
        "error": None,
        "error_at": None,
        "created_at": old,
        "updated_at": old,
    })

    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    # the file is present on disk
    monkeypatch.setattr(cheshire_cat.file_manager, "read_file", lambda source, remote: b"content")
    # ... and still listed, so the GC sweep keeps the completed entry
    monkeypatch.setattr(cheshire_cat.file_manager, "list_files", lambda path: [type("F", (), {"name": "sweep.pdf"})()])

    calls = []

    async def fake_ingest_file(cat, file, metadata, filename=None, **kwargs):
        calls.append((cat, file, filename))
        await ingestion_plugin.after_rabbithole_stored_documents.function(filename, [object()], cat)

    monkeypatch.setattr(lizard.rabbit_hole, "ingest_file", fake_ingest_file)

    # only this agent is enumerated by the startup pass
    monkeypatch.setattr(
        "cat.core_plugins.ingestion_status.resume.crud_settings.get_agents_main_keys",
        AsyncMock(return_value=[agent_key]),
    )
    monkeypatch.setenv("CAT_INGESTION_RESUME_INTERVAL_SECONDS", "60")

    # stop the loop after its first pass: the post-pass sleep raises CancelledError
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        if seconds >= 1:
            raise asyncio.CancelledError()
        await real_sleep(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    task = asyncio.create_task(resume._periodic_sweep_loop(lizard))
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(calls) == 1
    assert calls[0][2] == "sweep.pdf"
    doc = await get_status(agent_key, "agent", "sweep.pdf")
    assert doc is not None
    assert doc["status"] == "completed"


async def test_heartbeat_keeps_processing_fresh_and_blocks_double_resume(cheshire_cat, monkeypatch):
    """A ``processing`` entry with an active heartbeat stays fresh, and a second
    worker does NOT re-ingest it.

    ``_heartbeat_status`` bumps ``updated_at`` while the row is PROCESSING, so
    a long parse is never re-claimed as stale. Once the heartbeat has advanced
    ``updated_at``, ``_resume_agent`` sees a fresh entry and skips it (the
    per-source claim returns None) — the double-worker guard.
    """
    lizard = cheshire_cat.lizard
    agent_key = cheshire_cat.agent_key
    source = "heartbeat.pdf"

    old = time.time() - 1000
    await crud.store(status_key(agent_key, "agent", source), {
        "source": source,
        "scope": "agent",
        "chat_id": None,
        "type": "file",
        "status": "processing",
        "error": None,
        "error_at": None,
        "created_at": old,
        "updated_at": old,
    })

    # start the heartbeat with a short interval
    heartbeat_task = asyncio.create_task(
        ingestion_plugin._heartbeat_status(agent_key, "agent", source, interval=0.01)
    )

    # let a few beats run, then check the row was kept fresh
    await asyncio.sleep(0.05)

    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "processing"
    assert doc["updated_at"] > old

    # stop the heartbeat
    heartbeat_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await heartbeat_task

    # double-worker guard: the entry is now fresh, so a second _resume_agent
    # must NOT re-ingest it
    monkeypatch.setattr(lizard, "get_cheshire_cat", AsyncMock(return_value=cheshire_cat))
    monkeypatch.setattr(cheshire_cat.file_manager, "read_file", lambda source, remote: b"content")

    calls = []

    async def fake_ingest_file(cat, file, metadata, filename=None, **kwargs):
        calls.append(filename)

    monkeypatch.setattr(lizard.rabbit_hole, "ingest_file", fake_ingest_file)

    await resume._resume_agent(lizard, agent_key)

    assert calls == []
    doc = await get_status(agent_key, "agent", source)
    assert doc is not None
    assert doc["status"] == "processing"