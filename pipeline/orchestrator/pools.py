"""Bounded async resource pools for concurrency control in the pipeline orchestrator.
Pools gate concurrent worker access to named resources (e.g. gpu.0, cpu, io.Drive2)
and support runtime target adjustments without preempting in-flight tasks.
"""

import asyncio
from contextlib import asynccontextmanager


class ResourcePool:
    def __init__(self, name: str, target: int) -> None:
        self._name = name
        self._target = target
        self._in_use = 0
        self._waiting = 0
        self._condition = asyncio.Condition()

    @property
    def name(self) -> str:
        return self._name

    @property
    def target(self) -> int:
        return self._target

    @property
    def in_use(self) -> int:
        return self._in_use

    @property
    def waiting(self) -> int:
        return self._waiting

    @asynccontextmanager
    async def acquire(self):
        async with self._condition:
            self._waiting += 1
            try:
                # Re-check after every notify because target may have been lowered
                # while multiple waiters were already queued; a single notify_all
                # from set_target must not let more tasks through than the new target allows.
                await self._condition.wait_for(
                    lambda: self._in_use < self._target
                )
                self._in_use += 1
            finally:
                self._waiting -= 1

        try:
            yield
        finally:
            async with self._condition:
                self._in_use -= 1
                self._condition.notify_all()

    def set_target(self, n: int) -> None:
        if n < 0:
            raise ValueError(f"target must be >= 0, got {n}")

        async def _update() -> None:
            async with self._condition:
                self._target = n
                # Wake all waiters so they re-evaluate the (possibly raised) target.
                self._condition.notify_all()

        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_update())
        else:
            loop.run_until_complete(_update())


class PoolRegistry:
    def __init__(self) -> None:
        self._pools: dict[str, ResourcePool] = {}

    def register(self, name: str, target: int) -> ResourcePool:
        if name in self._pools:
            raise KeyError(f"Pool '{name}' is already registered")
        pool = ResourcePool(name, target)
        self._pools[name] = pool
        return pool

    def get(self, name: str) -> ResourcePool:
        return self._pools[name]

    def all(self) -> dict[str, ResourcePool]:
        return dict(self._pools)

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {
            name: {
                "target": pool.target,
                "in_use": pool.in_use,
                "waiting": pool.waiting,
            }
            for name, pool in self._pools.items()
        }
