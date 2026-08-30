"""Ingestion-status lifecycle hooks.

Observes the RabbitHole ingestion pipeline and writes per-source status docs
into Redis via the plugin's registry module.

Lifecycle written by these hooks:
    files:  uploaded -> processing -> completed | error
    urls:   uploaded -> downloading -> downloaded -> processing -> completed | error
"""
import asyncio

from cat import hook
from cat.core_plugins.ingestion_status.registry import (
    IngestionStatus,
    get_status,
    set_status,
)


def _scope_and_chat(cat):
    """Resolve ``(scope, chat_id)`` from the hook caller.

    A StrayCat carries a conversation ``id``; a CheshireCat does not, so the
    scope is the agent KB. Mirrors ``chat_id = self.stray.id if self.stray else None``.
    """
    if hasattr(cat, "id"):
        return cat.id, cat.id
    return "agent", None


def _source_type(source: str, is_url: bool = False) -> str:
    if is_url or source.startswith("http"):
        return "url"
    return "file"


async def _heartbeat_status(agent_id: str, scope: str, source: str, interval: float) -> None:
    """Periodically bump ``updated_at`` of a PROCESSING row while it is being handled.

    A long parse/embed can otherwise make the row look stale and get re-claimed
    by another worker (see ``claim_source_for_resume``). Stops as soon as the
    row leaves the PROCESSING state.
    """
    while True:
        await asyncio.sleep(interval)
        current = await get_status(agent_id, scope, source)
        if current and current.get("status") == IngestionStatus.PROCESSING.value:
            await set_status(
                agent_id,
                scope,
                source,
                type_=current.get("type", "file"),
                status=IngestionStatus.PROCESSING,
                chat_id=current.get("chat_id"),
            )
        else:
            return


@hook(priority=0)
async def rabbithole_ingestion_start(source, metadata, is_url, cat) -> None:
    """Record that an ingestion is about to begin (source known, nothing stored yet)."""
    scope, chat_id = _scope_and_chat(cat)
    await set_status(
        cat.agent_key,
        scope,
        source,
        type_=_source_type(source, is_url),
        status=IngestionStatus.UPLOADED,
        chat_id=chat_id,
    )


@hook(priority=0)
async def rabbithole_url_downloading(url, filename, cat) -> None:
    """Record that a URL download is about to start."""
    scope, chat_id = _scope_and_chat(cat)
    await set_status(
        cat.agent_key,
        scope,
        url,
        type_="url",
        status=IngestionStatus.DOWNLOADING,
        chat_id=chat_id,
    )


@hook(priority=0)
async def rabbithole_url_download_completed(url, filename, cat) -> None:
    """Record that a URL download completed successfully."""
    scope, chat_id = _scope_and_chat(cat)
    await set_status(
        cat.agent_key,
        scope,
        url,
        type_="url",
        status=IngestionStatus.DOWNLOADED,
        chat_id=chat_id,
    )


@hook(priority=0)
async def rabbithole_ingestion_processing(source, cat) -> None:
    """Record that the source is being embedded and stored."""
    scope, chat_id = _scope_and_chat(cat)
    await set_status(
        cat.agent_key,
        scope,
        source,
        type_=_source_type(source),
        status=IngestionStatus.PROCESSING,
        chat_id=chat_id,
    )


@hook(priority=0)
async def after_rabbithole_stored_documents(source, stored_points, cat) -> None:
    """Record that the source was stored successfully.

    The hook fires in the ``finally`` block of ``ingest_file``, so it also runs
    on the error path: never overwrite an already-recorded ERROR state, and
    ignore the unresolved empty source.
    """
    if not source:
        return
    scope, chat_id = _scope_and_chat(cat)
    current = await get_status(cat.agent_key, scope, source)
    if current and current.get("status") == IngestionStatus.ERROR.value:
        return
    await set_status(
        cat.agent_key,
        scope,
        source,
        type_=_source_type(source),
        status=IngestionStatus.COMPLETED,
        chat_id=chat_id,
    )


@hook(priority=0)
async def rabbithole_ingestion_error(source, error, cat) -> None:
    """Record that the ingestion failed, with the error message."""
    scope, chat_id = _scope_and_chat(cat)
    await set_status(
        cat.agent_key,
        scope,
        source,
        type_=_source_type(source),
        status=IngestionStatus.ERROR,
        chat_id=chat_id,
        error=str(error),
    )
