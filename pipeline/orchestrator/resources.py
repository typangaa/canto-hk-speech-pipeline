"""
Background sampler that polls GPU compute occupancy via nvidia-smi and adjusts
ResourcePool targets according to per-GPU yield/cap/exempt policies, so the
orchestrator coexists gracefully with foreign GPU jobs on the same machine.
"""

import asyncio
import logging
import subprocess
from collections.abc import Callable
from enum import Enum

from .pools import PoolRegistry

logger = logging.getLogger(__name__)

# GPU index inferred from pool name by stripping the "gpu." prefix, e.g. "gpu.0" -> 0.
_GPU_POOL_PREFIX = "gpu."


def _gpu_index_from_pool_name(name: str) -> int | None:
    """Return the integer GPU index encoded in a pool name, or None if unparseable."""
    if name.startswith(_GPU_POOL_PREFIX):
        suffix = name[len(_GPU_POOL_PREFIX):]
        if suffix.isdigit():
            return int(suffix)
    return None


def detect_foreign_gpu_pids(own_pids: set[int]) -> dict[int, set[int]]:
    """Return a mapping of GPU index -> set of *foreign* compute PIDs.

    Shells out to nvidia-smi twice: once to build a uuid->index map, once to
    list all active compute PIDs with their GPU uuids.  own_pids are subtracted
    so that the orchestrator's own workers are never treated as foreign.

    Returns {} (empty dict, not an empty-per-GPU dict) on any error (nvidia-smi
    absent, non-zero exit, or unparseable output) so callers can distinguish
    "no foreign pids found" from "we couldn't check".
    """
    # Step 1: build uuid -> index map.
    try:
        uuid_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        logger.debug("nvidia-smi not found; skipping GPU foreign-pid detection")
        return {}
    except Exception as exc:
        logger.warning("nvidia-smi (uuid query) raised unexpectedly: %s", exc)
        return {}

    if uuid_result.returncode != 0:
        logger.warning(
            "nvidia-smi --query-gpu exited %d: %s",
            uuid_result.returncode,
            uuid_result.stderr.strip(),
        )
        return {}

    uuid_to_index: dict[str, int] = {}
    for line in uuid_result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            continue
        idx_str, uuid = parts
        if idx_str.isdigit():
            uuid_to_index[uuid] = int(idx_str)

    if not uuid_to_index:
        logger.warning("nvidia-smi uuid query returned no parseable rows")
        return {}

    # Initialise result with every known GPU index mapped to an empty set so
    # callers can iterate all known GPUs even when no foreign pids are present.
    result: dict[int, set[int]] = {idx: set() for idx in uuid_to_index.values()}

    # Step 2: list active compute apps.
    try:
        apps_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        logger.warning("nvidia-smi (compute-apps query) raised unexpectedly: %s", exc)
        return {}

    if apps_result.returncode != 0:
        logger.warning(
            "nvidia-smi --query-compute-apps exited %d: %s",
            apps_result.returncode,
            apps_result.stderr.strip(),
        )
        return {}

    for line in apps_result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            continue
        pid_str, uuid = parts
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        gpu_idx = uuid_to_index.get(uuid)
        if gpu_idx is None:
            continue
        # Only record pids that do not belong to our own workers.
        if pid not in own_pids:
            result[gpu_idx].add(pid)

    return result


class GpuPolicy(str, Enum):
    """Per-GPU pool scheduling policy when a foreign compute process is detected."""

    YIELD = "yield"   # Set pool target to 0 while foreign pids are present.
    CAP = "cap"       # Clamp pool target to cap_target while foreign pids are present.
    EXEMPT = "exempt" # Never touch this pool's target regardless of foreign pids.


class Sampler:
    """Background asyncio task that polls GPU occupancy and adjusts pool targets.

    Caller pattern (cancellable-task approach):
        task = asyncio.create_task(sampler.run())
        ...
        task.cancel()          # or call sampler.stop() which does the same
        await asyncio.gather(task, return_exceptions=True)

    No EMA smoothing is applied to the foreign-pid signal; pid presence is
    binary (a training job either occupies the GPU or it doesn't) so smoothing
    would only delay the response.  EMA smoothing for the CPU/IO load-average
    signal is handled by a separate module not yet implemented.
    """

    def __init__(
        self,
        registry: PoolRegistry,
        gpu_policies: dict[str, GpuPolicy],
        own_pids: Callable[[], set[int]],
        poll_interval: float = 2.0,
        cap_target: int = 1,
    ) -> None:
        """
        Args:
            registry:      Pool registry used to look up managed pools.
            gpu_policies:  Maps pool name (e.g. "gpu.0") -> GpuPolicy.
                           Pools absent from this mapping are never touched.
            own_pids:      Callable returning the current set of orchestrator
                           worker PIDs.  Called fresh on every poll because the
                           worker set changes dynamically as tasks spawn and exit;
                           a fixed snapshot taken at __init__ time would produce
                           false-positive foreign-pid detections for newly spawned
                           workers that joined after construction.
            poll_interval: Seconds between successive nvidia-smi polls.
            cap_target:    Pool target used by the CAP policy when foreign pids
                           are detected.
        """
        self._registry = registry
        self._gpu_policies = gpu_policies
        self._own_pids = own_pids
        self._poll_interval = poll_interval
        self._cap_target = cap_target

        # Snapshot the *original* configured targets at construction time.
        # set_target() mutates pool.target, so we cannot recover the original
        # value later without storing it here.  Absence from this dict means the
        # pool was not registered at construction time and will not be managed.
        self._original_targets: dict[str, int] = {}
        for pool_name in gpu_policies:
            try:
                pool = registry.get(pool_name)
                self._original_targets[pool_name] = pool.target
            except KeyError:
                logger.warning(
                    "Pool %r listed in gpu_policies but not found in registry; "
                    "it will be skipped during sampling.",
                    pool_name,
                )

        self._stop_event = asyncio.Event()

    def stop(self) -> None:
        """Signal the run() loop to exit cleanly after the current sleep."""
        self._stop_event.set()

    async def run(self) -> None:
        """Poll nvidia-smi and adjust pool targets indefinitely until stopped.

        Designed to be launched as an asyncio.Task.  The loop is also cleanly
        terminated by task.cancel() in addition to stop().
        """
        logger.info(
            "Sampler starting; poll_interval=%.1fs, managed pools=%s",
            self._poll_interval,
            list(self._gpu_policies.keys()),
        )

        while not self._stop_event.is_set():
            # Use wait_for on the stop event so that stop() wakes us immediately
            # rather than having to wait out the remainder of the sleep interval.
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval,
                )
                # If we reach here the event fired; exit the loop.
                break
            except asyncio.TimeoutError:
                pass  # Normal path: interval elapsed, proceed to sample.

            await self._sample_once()

        logger.info("Sampler stopped.")

    async def _sample_once(self) -> None:
        """Single poll: detect foreign pids, apply policies to managed pools."""
        try:
            # Run the blocking subprocess call in the default thread-pool executor
            # so we do not stall the event loop during nvidia-smi's latency.
            loop = asyncio.get_running_loop()
            current_own_pids: set[int] = self._own_pids()
            foreign_by_gpu: dict[int, set[int]] = await loop.run_in_executor(
                None, detect_foreign_gpu_pids, current_own_pids
            )
        except Exception as exc:
            logger.warning("Unexpected error during GPU pid detection: %s", exc)
            return

        if not foreign_by_gpu:
            # detect_foreign_gpu_pids returns {} on any nvidia-smi error;
            # we skip policy application rather than incorrectly restoring
            # targets based on absent data.
            logger.debug("GPU pid detection returned empty result; skipping policy step.")
            return

        for pool_name, policy in self._gpu_policies.items():
            if pool_name not in self._original_targets:
                continue  # Pool was absent from registry at init; skip.

            if policy is GpuPolicy.EXEMPT:
                # EXEMPT pools are never touched, even if a foreign pid is present.
                continue

            gpu_idx = _gpu_index_from_pool_name(pool_name)
            if gpu_idx is None:
                logger.warning(
                    "Cannot parse GPU index from pool name %r; skipping.", pool_name
                )
                continue

            try:
                pool = self._registry.get(pool_name)
            except KeyError:
                logger.warning("Pool %r disappeared from registry; skipping.", pool_name)
                continue

            foreign_pids = foreign_by_gpu.get(gpu_idx, set())
            original = self._original_targets[pool_name]

            if foreign_pids:
                if policy is GpuPolicy.YIELD:
                    if pool.target != 0:
                        logger.info(
                            "GPU %d: foreign pids %s detected; YIELD -> setting %r target to 0 (was %d)",
                            gpu_idx, foreign_pids, pool_name, pool.target,
                        )
                        pool.set_target(0)
                elif policy is GpuPolicy.CAP:
                    if pool.target != self._cap_target:
                        logger.info(
                            "GPU %d: foreign pids %s detected; CAP -> setting %r target to %d (was %d)",
                            gpu_idx, foreign_pids, pool_name, self._cap_target, pool.target,
                        )
                        pool.set_target(self._cap_target)
            else:
                # No foreign pids on this GPU: restore to original target.
                # We always restore (not just lower) because a previous poll may
                # have suppressed the target and we must undo that when the
                # foreign job exits.  The orchestrator's own scaling logic
                # (outside this module) is responsible for any further caps.
                if pool.target != original:
                    logger.info(
                        "GPU %d: no foreign pids; restoring %r target to %d (was %d)",
                        gpu_idx, pool_name, original, pool.target,
                    )
                    pool.set_target(original)
