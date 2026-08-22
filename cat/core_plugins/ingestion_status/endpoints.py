"""``GET /ingestion/status`` endpoint with read-time reconcile.

Exposes the ingestion-status registry over HTTP. Before returning, the
registry is reconciled against the canonical source lists (files on disk via
the file manager, URLs via the web points in the vector store) and entries
whose source no longer exists are purged (lazy delete). Purge failures are
logged and never fail the request.
"""
import os
from typing import Dict, List, Optional

from fastapi import Query

from cat import (
    AuthorizedInfo,
    AuthPermission,
    AuthResource,
    check_permissions,
    endpoint,
)
from cat.core_plugins.ingestion_status.registry import delete_status, list_statuses
from cat.db.cruds import conversations as crud_conversations
from cat.log import log
from cat.services.memory.models import VectorMemoryType


async def _canonical_files(ccat, path: str) -> set:
    """Canonical file names for a scope path (agent or chat)."""
    try:
        return {f.name for f in ccat.file_manager.list_files(path)}
    except Exception as e:
        log.error(f"Ingestion status reconcile: failed to list files for {path}: {e}")
        return set()


async def _canonical_urls(ccat, collection, chat_id: Optional[str]) -> set:
    """Canonical URL sources for a scope, derived from the web points."""
    try:
        # the Qdrant gRPC client needs the collection name as a str, not the
        # VectorMemoryType enum (mirror reconcile.py which does str(...))
        memory_points, _ = await ccat.vector_memory_handler.get_all_tenant_points_from_web(str(collection))
    except Exception as e:
        log.error(f"Ingestion status reconcile: failed to list web points for {collection}: {e}")
        return set()

    result = set()
    for memory_point in memory_points:
        payload = memory_point.payload or {}
        metadata = payload.get("metadata", {})
        if chat_id is not None and metadata.get("chat_id") != chat_id:
            continue
        source = metadata.get("source")
        if source and source.startswith("http"):
            result.add(source)
    return result


async def _reconcile(ccat, agent_id: str, chat_id: Optional[str], statuses: List[Dict]) -> List[Dict]:
    """Purge registry entries whose source is absent from the canonical lists.

    Returns the reconciled (post-purge) list. Purge failures are logged and
    never propagated to the caller.

    ERROR carve-out: entries whose ``status`` is ``"error"`` are never purged
    by this canonical-source reconcile. A failed upload never lands in the file
    manager / vector store, so it would otherwise be purged on first read and
    the error badge could never be shown. Error entries remain observable via
    the API (including chat-scoped errors whose conversation is gone — the
    error is still meaningful). They are only removed by whole-scope wipes
    (``clear_agent`` / ``clear_chat`` / destroy).
    """
    scope = chat_id or "agent"

    if chat_id is None:
        file_path = ccat.agent_key
        collection = VectorMemoryType.DECLARATIVE
    else:
        file_path = os.path.join(ccat.agent_key, chat_id)
        collection = VectorMemoryType.EPISODIC

    canonical_files = await _canonical_files(ccat, file_path)
    canonical_urls = await _canonical_urls(ccat, collection, chat_id)

    # chat-scoped entries additionally require the conversation to still exist
    chat_exists = True
    if chat_id is not None:
        chat_exists = (
            await crud_conversations.get_user_id_from_conversation_keys(agent_id, chat_id)
            is not None
        )

    kept: List[Dict] = []
    for doc in statuses:
        source = doc.get("source")
        if not source:
            # cannot reconcile without a source: keep it
            kept.append(doc)
            continue

        # ERROR carve-out: never purge error entries via the canonical reconcile
        if doc.get("status") == "error":
            kept.append(doc)
            continue

        type_ = doc.get("type")

        if type_ == "file":
            exists = source in canonical_files
        elif type_ == "url":
            exists = source in canonical_urls
        else:
            exists = True  # unknown type: keep

        if chat_id is not None and not chat_exists:
            exists = False

        if not exists:
            try:
                await delete_status(agent_id, scope, source)
            except Exception as e:
                log.error(f"Ingestion status reconcile: failed to purge {source}: {e}")
            continue

        kept.append(doc)

    return kept


@endpoint.get("/status", prefix="/ingestion", tags=["Ingestion"])
async def get_ingestion_status(
    chat_id: Optional[str] = Query(
        default=None,
        description="Conversation id to scope the statuses to a single chat.",
    ),
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.READ),
) -> List[Dict]:
    """Return the ingestion-status registry, reconciled against canonical sources.

    Agent scope by default; pass ``?chat_id=<id>`` for a conversation scope.
    Entries whose source no longer exists (file removed, URL without web
    points, or conversation gone) are purged before returning.
    """
    agent_id = info.agent_id
    if agent_id is None:
        return []

    statuses = await list_statuses(agent_id, chat_id)

    if info.cheshire_cat is None:
        return statuses

    return await _reconcile(info.cheshire_cat, agent_id, chat_id, statuses)