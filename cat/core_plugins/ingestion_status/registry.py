"""Redis-backed ingestion-status registry for the ``ingestion_status`` core plugin.

Key namespace (owned by this plugin; consumed by the destroy purge and the
re-embed flow): ``agents:{agent_id}:ingestion:{scope}:{sha256(source)}`` where
``scope`` is ``"agent"`` for agent-KB ingestion or a conversation ``chat_id``.

Every status transition is written as a Redis-JSON document via ``cat.db.crud``
(no raw Redis client is used here).
"""
import hashlib
from typing import Dict, List, Optional

from cat.db import crud
from cat.db.database import get_async_db
from cat.db.models import generate_timestamp
from cat.utils import Enum


class IngestionStatus(Enum):
    """Lifecycle states of a source ingestion.

    Files: ``uploaded -> processing -> completed | error``.
    URLs add ``downloading -> downloaded`` between ``uploaded`` and ``processing``.
    """
    UPLOADED = "uploaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


def status_key(agent_id: str, scope: str, source: str) -> str:
    """Redis key for a source's ingestion status.

    Args:
        agent_id: The agent (chatbot) id.
        scope: ``"agent"`` for the agent KB, or a conversation ``chat_id``.
        source: The ingested file name or URL.

    Returns:
        ``agents:{agent_id}:ingestion:{scope}:{sha256(source)}``
    """
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return f"agents:{agent_id}:ingestion:{scope}:{digest}"


async def _read_doc(key: str) -> Optional[Dict]:
    """Read a status doc, unwrapping the RedisJSON array wrapper."""
    value = await crud.read(key)
    if value is None:
        return None
    if isinstance(value, list):
        return value[0] if value else None
    return value


async def set_status(
    agent_id: str,
    scope: str,
    source: str,
    *,
    type_: str,
    status: IngestionStatus,
    chat_id: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict:
    """Write (or update) the ingestion-status doc for a source.

    ``created_at`` is preserved across updates; ``updated_at`` is bumped on
    every write; ``error_at`` is set when ``status`` is ERROR.

    Args:
        agent_id: The agent (chatbot) id.
        scope: ``"agent"`` for the agent KB, or a conversation ``chat_id``.
        source: The ingested file name or URL.
        type_: ``"file"`` or ``"url"``.
        status: The new lifecycle state.
        chat_id: The conversation id, when the ingestion is chat-scoped.
        error: The error message, when ``status`` is ERROR.

    Returns:
        The stored status document.
    """
    key = status_key(agent_id, scope, source)
    now = generate_timestamp()

    existing = await _read_doc(key)
    created_at = existing.get("created_at", now) if existing else now

    doc = {
        "source": source,
        "scope": scope,
        "chat_id": chat_id,
        "type": type_,
        "status": status,
        "error": error,
        "error_at": now if status == IngestionStatus.ERROR else None,
        "created_at": created_at,
        "updated_at": now,
    }
    await crud.store(key, doc)
    return doc


async def get_status(agent_id: str, scope: str, source: str) -> Optional[Dict]:
    """Read the ingestion-status doc for a source, or None if absent."""
    return await _read_doc(status_key(agent_id, scope, source))


async def delete_status(agent_id: str, scope: str, source: str) -> None:
    """Delete the ingestion-status doc for a source."""
    await crud.delete(status_key(agent_id, scope, source))


async def list_statuses(agent_id: str, chat_id: Optional[str] = None) -> List[Dict]:
    """List the ingestion-status docs for an agent.

    With no ``chat_id`` only agent-scope entries are returned; with a
    ``chat_id`` only that conversation's entries are returned.
    """
    db = get_async_db()
    results: List[Dict] = []
    async for key in db.scan_iter(f"agents:{agent_id}:ingestion:*"):
        doc = await _read_doc(key)
        if not doc:
            continue
        scope = doc.get("scope")
        if chat_id is None:
            if scope != "agent":
                continue
        elif scope != chat_id:
            continue
        results.append(doc)
    return results


async def clear_agent(agent_id: str) -> int:
    """Delete every ingestion-status key for an agent (all scopes).

    Returns:
        The number of keys deleted.
    """
    return await crud.destroy(f"agents:{agent_id}:ingestion:*")


async def clear_chat(agent_id: str, chat_id: str) -> int:
    """Delete every ingestion-status key for one conversation scope.

    Returns:
        The number of keys deleted.
    """
    return await crud.destroy(f"agents:{agent_id}:ingestion:{chat_id}:*")