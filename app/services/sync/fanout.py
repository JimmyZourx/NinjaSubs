"""Process-wide concurrency gate for reference-provider fan-out."""

from __future__ import annotations

import asyncio
import threading
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.config import settings


class ReferenceFanoutLimiter:
    """Bound admitted lookups per event loop before they create provider tasks."""

    def __init__(self, limit: int | None = None) -> None:
        configured = limit if limit is not None else settings.REFERENCE_RESOLUTION_MAX_CONCURRENCY
        self.limit = max(1, int(configured))
        self._semaphores: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

    def _semaphore_for_loop(self, loop: asyncio.AbstractEventLoop) -> asyncio.Semaphore:
        # Weak loop keys prevent test/application loop lifetimes from accumulating.
        with self._lock:
            semaphore = self._semaphores.get(loop)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self.limit)
                self._semaphores[loop] = semaphore
            return semaphore

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        semaphore = self._semaphore_for_loop(loop)
        async with semaphore:
            yield


# All ExternalExactStrategy instances share this process-level limiter.
reference_fanout_limiter = ReferenceFanoutLimiter()
