"""Dedicated low-concurrency executor lane for heavy ingestion work.

Ingestion (chunking, embedding, storing) is CPU/IO heavy and, when run on the
shared default ``ThreadPoolExecutor``, can saturate the pool that chat-time
embedding/chunking also relies on, degrading interactive latency.

This module provides a process-wide, lazily-created ``ThreadPoolExecutor``
whose worker count is bounded by ``CAT_INGESTION_WORKERS``. It complements the
``CAT_INGESTION_MAX_CONCURRENCY`` semaphore in ``cat.rabbit_hole``: the
semaphore bounds how many ingestion tasks run at once, while this pool keeps
those tasks off the shared default executor.

Importing this module has zero side effects: the pool is only created on first
use via ``_get_ingestion_pool()``.
"""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from cat.env import get_env_int

T = TypeVar("T")

_pool: ThreadPoolExecutor | None = None


def _get_ingestion_pool() -> ThreadPoolExecutor | None:
    """Return the process-wide ingestion pool, creating it on first use.

    The worker count is read at runtime from ``CAT_INGESTION_WORKERS`` and the
    pool is then cached for the process lifetime. ``None`` or ``<= 0`` means
    the lane is disabled: no pool is created and callers fall back to the
    default executor (previous behavior is preserved).
    """
    global _pool
    if _pool is None:
        max_workers = get_env_int("CAT_INGESTION_WORKERS")
        if max_workers is None or max_workers <= 0:
            return None
        _pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="cat-ingestion",
        )
    return _pool


async def run_in_ingestion_executor(
    func: Callable[..., T], *args, **kwargs
) -> T:
    """Run ``func(*args, **kwargs)`` on the dedicated ingestion pool.

    If the ingestion lane is disabled (``CAT_INGESTION_WORKERS`` unset or
    ``<= 0``), the call is dispatched to the default executor, preserving the
    prior behavior.
    """
    loop = asyncio.get_running_loop()
    pool = _get_ingestion_pool()
    if pool is None:
        return await loop.run_in_executor(None, func, *args, **kwargs)
    return await loop.run_in_executor(pool, func, *args, **kwargs)