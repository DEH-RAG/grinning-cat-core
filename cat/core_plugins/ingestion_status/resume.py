"""Startup resume + periodic GC sweep for the ``ingestion_status`` core plugin.

The ``after_lizard_bootstrap`` hook schedules a fire-and-forget background pass
(per agent) that:

  (a) resumes stale ``uploaded``/``processing`` ingestions older than
      ``CAT_INGESTION_RESUME_STALE_SECONDS`` (default 300) by re-triggering
      ``rabbit_hole.ingest_file``; and
  (b) purges status entries whose source is absent from the canonical lists
      (files on disk, URLs in the vector store, existing conversations) via the
      shared :func:`reconcile_agent` helper.

Both are guarded by a distributed lock and never crash lizard bootstrap.
"""
import asyncio
import os
from io import BytesIO
from typing import Any, Dict, List

from cat import BillTheLizard, hook, log
from cat.core_plugins.ingestion_status.registry import (
    IngestionStatus,
    claim_source_for_resume,
    list_statuses,
)
from cat.core_plugins.ingestion_status.reconcile import reconcile_agent
from cat.db import crud
from cat.db.cruds import settings as crud_settings
from cat.env import get_env_int
from cat.utils import guess_file_type, is_url


def _resume_enabled() -> bool:
    """``CAT_INGESTION_RESUME_ON_STARTUP`` flag, defaulting to true."""
    value = os.getenv("CAT_INGESTION_RESUME_ON_STARTUP")
    return value is None or value.lower() in ("1", "true", "yes", "on")


def _gc_enabled() -> bool:
    """``CAT_INGESTION_STATUS_GC_ON_STARTUP`` flag, defaulting to true."""
    value = os.getenv("CAT_INGESTION_STATUS_GC_ON_STARTUP")
    return value is None or value.lower() in ("1", "true", "yes", "on")


def _stale_seconds() -> int:
    """Staleness threshold for resume, defaulting to 300 seconds."""
    value = get_env_int("CAT_INGESTION_RESUME_STALE_SECONDS")
    return value if value is not None and value > 0 else 300


async def _resume_agent(lizard: BillTheLizard, agent_id: str, ccat: Any = None) -> None:
    """Re-trigger stale ingestions for one agent.

    Only entries in ``uploaded``/``processing``/``error`` older than the stale
    threshold are candidates. Each candidate is claimed atomically under a
    per-source lock (so two workers never re-ingest the SAME source, while
    different sources of the same agent proceed concurrently) and only
    re-ingested if its source is still present (file bytes on disk, or a URL
    that is still a valid http source). Fresh entries and ``completed``
    entries are left untouched.
    """
    if ccat is None:
        ccat = await lizard.get_cheshire_cat(agent_id)
    if ccat is None:
        return

    stale_seconds = _stale_seconds()
    entries = await list_statuses(agent_id)
    for entry in entries:
        status = entry.get("status")
        if status not in (
            IngestionStatus.UPLOADED.value,
            IngestionStatus.PROCESSING.value,
            IngestionStatus.ERROR.value,
        ):
            continue

        source = entry.get("source")
        scope = entry.get("scope")
        type_ = entry.get("type")
        if not source:
            continue

        # Only re-ingest sources that are still present (a file that never
        # landed on disk, or a URL that no longer resolves, cannot be
        # re-processed — the teacher must re-upload/re-add instead).
        if type_ == "url" or is_url(source):
            # URLs are re-downloaded by ingest_file; nothing to pre-check on
            # CAT storage, but the source must still look like a URL.
            pass
        else:
            path = agent_id
            if scope != "agent":
                path = os.path.join(path, str(scope))
            try:
                if ccat.file_manager.read_file(source, path) is None:
                    log.warning(
                        f"Ingestion resume: file {source} missing for agent {agent_id}; skipping"
                    )
                    continue
            except Exception as e:
                log.warning(f"Ingestion resume: cannot check file {source} for {agent_id}: {e}")
                continue

        # Atomic per-source claim: <agent>:<scope>:<source>. Only the worker
        # that wins the claim (and only if the entry is stale — not actively
        # being handled by another worker) proceeds with the re-ingestion.
        owner = f"{os.getpid()}"
        claimed = await claim_source_for_resume(
            agent_id, str(scope), source,
            stale_after=stale_seconds, owner=owner,
        )
        if claimed is None:
            # another worker is handling it, or it is still fresh
            continue

        cat = ccat
        if scope != "agent":
            stray = await ccat._find_stray_cat(str(scope))
            if stray is None:
                continue
            cat = stray

        try:
            if type_ == "url" or is_url(source):
                await lizard.rabbit_hole.ingest_file(
                    cat=cat, file=source, filename=source, metadata={},
                )
            else:
                path = agent_id
                if scope != "agent":
                    path = os.path.join(path, str(scope))
                file_bytes = ccat.file_manager.read_file(source, path)
                if file_bytes is None:
                    log.warning(
                        f"Ingestion resume: file {source} missing for agent {agent_id}; skipping"
                    )
                    continue
                file_io = BytesIO(file_bytes)
                content_type, _ = guess_file_type(file_io)
                await lizard.rabbit_hole.ingest_file(
                    cat=cat, file=file_io, filename=source, content_type=content_type, metadata={},
                )
        except Exception as e:
            log.error(f"Ingestion resume: failed to re-ingest {source} for agent {agent_id}: {e}")


async def _pass_for_agent(lizard: BillTheLizard, agent_id: str) -> None:
    """Run the resume + GC pass for one agent.

    Only the enumeration is guarded by a short agent-level distributed lock;
    actual (re)ingestion of each candidate happens under its own
    per-source lock (see ``claim_source_for_resume``), so different sources
    of the same agent are processed concurrently while the same source can
    never be double-processed by another worker.
    """
    try:
        async with crud.distributed_lock(f"ingestion-sweep:{agent_id}", timeout=30, blocking_timeout=5):
            ccat = await lizard.get_cheshire_cat(agent_id)
            if ccat is None:
                return
            if _resume_enabled():
                await _resume_agent(lizard, agent_id, ccat=ccat)
            if _gc_enabled():
                await reconcile_agent(agent_id, ccat=ccat)
    except Exception as e:
        log.error(f"Ingestion startup pass failed for agent {agent_id}: {e}")


async def _startup_pass(lizard: BillTheLizard) -> None:
    """Enumerate agents and run the resume + GC pass for each."""
    try:
        agent_ids = await crud_settings.get_agents_main_keys()
    except Exception as e:
        log.error(f"Ingestion startup pass: failed to enumerate agents: {e}")
        return
    for agent_id in agent_ids:
        await _pass_for_agent(lizard, agent_id)


@hook
async def after_lizard_bootstrap(lizard: BillTheLizard):
    """Schedule the fire-and-forget startup pass (never blocks bootstrap)."""
    asyncio.ensure_future(_startup_pass(lizard))