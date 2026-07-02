"""
Supervisor-side of the JSONL worker protocol.
Spawns worker subprocesses and communicates via newline-delimited JSON over stdin/stdout.
All messages are single-line JSON objects; stdout EOF signals subprocess exit.
"""

import asyncio
import json
import logging
import signal

log = logging.getLogger(__name__)


class WorkerDiedError(Exception):
    pass


class WorkerHandle:
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process

    @property
    def pid(self) -> int:
        return self._process.pid

    async def send_task(self, task_id: str, items: list[dict]) -> None:
        msg = json.dumps({"type": "task", "task_id": task_id, "items": items})
        self._process.stdin.write((msg + "\n").encode())
        await self._process.stdin.drain()

    async def read_message(self, timeout: float | None = None) -> dict:
        async def _read() -> dict:
            line = await self._process.stdout.readline()
            # readline() returning b"" is not an empty JSONL line — it means the
            # read end of the pipe is closed because the process has exited.
            if not line:
                stderr_tail = await self._read_stderr_tail()
                raise WorkerDiedError(
                    f"Worker pid={self.pid} exited unexpectedly. "
                    f"stderr tail:\n{stderr_tail}"
                )
            return json.loads(line)

        if timeout is not None:
            return await asyncio.wait_for(_read(), timeout=timeout)
        return await _read()

    async def wait_ready(self, timeout: float = 30.0) -> dict:
        msg = await self.read_message(timeout=timeout)
        if msg.get("type") != "ready":
            raise WorkerDiedError(
                f"Expected 'ready' from worker pid={self.pid}, got: {msg!r}"
            )
        return msg

    async def shutdown(self, grace_period: float = 10.0) -> None:
        if self._process.returncode is not None:
            return

        payload = json.dumps({"type": "shutdown"})
        try:
            self._process.stdin.write((payload + "\n").encode())
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            # Process already gone; fall through to wait/kill logic.
            pass

        try:
            await asyncio.wait_for(self._process.wait(), timeout=grace_period)
            return
        except asyncio.TimeoutError:
            pass

        log.warning("Worker pid=%d did not exit after shutdown; sending SIGTERM", self.pid)
        try:
            self._process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return

        # SIGTERM is advisory — a misbehaving or OOM-killed process may ignore it,
        # so we bound the wait and escalate to SIGKILL to guarantee reclamation.
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
            return
        except asyncio.TimeoutError:
            pass

        log.error("Worker pid=%d did not respond to SIGTERM; sending SIGKILL", self.pid)
        await self.kill()

    async def kill(self) -> None:
        if self._process.returncode is not None:
            return
        try:
            self._process.kill()
        except ProcessLookupError:
            return
        await self._process.wait()

    async def _read_stderr_tail(self, max_bytes: int = 2000) -> str:
        if self._process.stderr is None:
            return "<no stderr>"
        try:
            data = await asyncio.wait_for(
                self._process.stderr.read(max_bytes), timeout=2.0
            )
            return data.decode(errors="replace").strip()
        except (asyncio.TimeoutError, Exception):
            return "<could not read stderr>"


async def spawn_worker(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
) -> WorkerHandle:
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    log.debug("Spawned worker pid=%d cmd=%r", process.pid, cmd)
    return WorkerHandle(process)
