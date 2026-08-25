import asyncio
import os

from cat.services import ingestion_executor


# Sample the effective nice of a worker thread by running a callback inside the pool.
def _sample_worker_niceness() -> int:
    pool = ingestion_executor._get_ingestion_pool()
    assert pool is not None
    return pool.submit(os.getpriority, os.PRIO_PROCESS, 0).result()


async def test_pool_workers_are_de_prioritized_by_default(monkeypatch):
    """Worker threads get a positive ``nice`` so they yield CPU under pressure.

    With the default ``CAT_INGESTION_NICENESS``, ``_get_ingestion_pool`` builds a
    ``ThreadPoolExecutor`` whose threads run at a lower scheduler priority than the
    main thread.
    """
    # force a fresh pool so any env override below is read
    monkeypatch.setattr(ingestion_executor, "_pool", None)
    monkeypatch.delenv("CAT_INGESTION_NICENESS", raising=False)
    monkeypatch.setenv("CAT_INGESTION_WORKERS", "1")

    pool = ingestion_executor._get_ingestion_pool()
    assert pool is not None

    main_nice = os.getpriority(os.PRIO_PROCESS, 0)
    worker_nice = await asyncio.to_thread(_sample_worker_niceness)
    assert worker_nice > main_nice


async def test_pool_niceness_disabled_when_zero(monkeypatch):
    """``CAT_INGESTION_NICENESS=0`` disables de-prioritization (no-op initializer).

    The initializer is still installed but self-guards on ``niceness <= 0``, so
    worker threads inherit the main thread's niceness.
    """
    monkeypatch.setattr(ingestion_executor, "_pool", None)
    monkeypatch.setenv("CAT_INGESTION_NICENESS", "0")
    monkeypatch.setenv("CAT_INGESTION_WORKERS", "1")

    pool = ingestion_executor._get_ingestion_pool()
    assert pool is not None

    main_nice = os.getpriority(os.PRIO_PROCESS, 0)
    worker_nice = await asyncio.to_thread(_sample_worker_niceness)
    assert worker_nice == main_nice


def test_get_ingestion_pool_returns_none_when_workers_non_positive(monkeypatch):
    """``CAT_INGESTION_WORKERS <= 0`` disables the lane (default executor fallback)."""
    monkeypatch.setattr(ingestion_executor, "_pool", None)
    monkeypatch.setenv("CAT_INGESTION_WORKERS", "0")
    assert ingestion_executor._get_ingestion_pool() is None