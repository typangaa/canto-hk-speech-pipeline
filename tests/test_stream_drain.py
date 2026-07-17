import types

from pipeline.tools import stream_drain
from pipeline.tools.stream_drain import run_stream


class FakePopen:
    """Simulates subprocess.Popen: poll() consumes a scripted sequence,
    then holds at the last value; wait() returns wait_returncode."""

    def __init__(self, poll_sequence, wait_returncode=0):
        self._sequence = list(poll_sequence)
        self._i = 0
        self._wait_returncode = wait_returncode
        self.cmd = None

    def poll(self):
        if self._i < len(self._sequence):
            v = self._sequence[self._i]
            self._i += 1
            return v
        return self._sequence[-1] if self._sequence else self._wait_returncode

    def wait(self):
        return self._wait_returncode


def _fake_completed(returncode=0, stdout="ok"):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout)


def _no_sleep(seconds):
    pass


def test_run_stream_launches_upstream_and_final_drains_when_immediately_done(monkeypatch):
    launched = {}

    def fake_launch(node, extra_args):
        launched["node"] = node
        launched["extra_args"] = extra_args
        return FakePopen(poll_sequence=[0], wait_returncode=0)  # exits immediately

    drained = []

    def fake_drain(nodes, extra_args):
        drained.append(list(nodes))
        return _fake_completed()

    monkeypatch.setattr(stream_drain, "_launch_upstream", fake_launch)
    monkeypatch.setattr(stream_drain, "_drain_downstream", fake_drain)

    result = run_stream(
        upstream="asr.transcribe",
        downstream=["asr.agreement"],
        poll_interval_s=1,
        sleep_fn=_no_sleep,
    )

    assert launched["node"] == "asr.transcribe"
    assert result["upstream_returncode"] == 0
    assert result["polls"] == []  # never entered the while-alive loop
    assert drained == [["asr.agreement"]]  # exactly one final drain


def test_run_stream_polls_while_upstream_alive_then_final_drains(monkeypatch):
    # None, None, None, None, 0 -> exactly 2 polls before exit (see FakePopen trace)
    monkeypatch.setattr(
        stream_drain, "_launch_upstream",
        lambda node, args: FakePopen(poll_sequence=[None, None, None, None, 0], wait_returncode=0),
    )
    drain_calls = []

    def fake_drain(nodes, extra_args):
        drain_calls.append(list(nodes))
        return _fake_completed()

    monkeypatch.setattr(stream_drain, "_drain_downstream", fake_drain)

    result = run_stream(
        upstream="asr.transcribe",
        downstream=["asr.agreement"],
        poll_interval_s=1,
        sleep_fn=_no_sleep,
    )

    assert len(result["polls"]) == 2
    # 2 mid-run polls + 1 final drain = 3 total drain calls
    assert len(drain_calls) == 3


def test_run_stream_passes_upstream_args(monkeypatch):
    captured = {}

    def fake_launch(node, extra_args):
        captured["extra_args"] = extra_args
        return FakePopen(poll_sequence=[0], wait_returncode=0)

    monkeypatch.setattr(stream_drain, "_launch_upstream", fake_launch)
    monkeypatch.setattr(stream_drain, "_drain_downstream", lambda nodes, extra_args: _fake_completed())

    run_stream(
        upstream="asr.transcribe",
        upstream_args=["--batch", "64"],
        downstream=["asr.agreement"],
        sleep_fn=_no_sleep,
    )
    assert captured["extra_args"] == ["--batch", "64"]


def test_run_stream_multi_downstream_passed_through_as_list(monkeypatch):
    monkeypatch.setattr(
        stream_drain, "_launch_upstream",
        lambda node, args: FakePopen(poll_sequence=[0], wait_returncode=0),
    )
    drained = []

    def fake_drain(nodes, extra_args):
        drained.append(list(nodes))
        return _fake_completed()

    monkeypatch.setattr(stream_drain, "_drain_downstream", fake_drain)

    run_stream(
        upstream="asr.transcribe",
        downstream=["asr.agreement", "g2p"],
        sleep_fn=_no_sleep,
    )
    assert drained == [["asr.agreement", "g2p"]]


def test_run_stream_nonzero_upstream_returncode_propagates(monkeypatch):
    monkeypatch.setattr(
        stream_drain, "_launch_upstream",
        lambda node, args: FakePopen(poll_sequence=[1], wait_returncode=1),
    )
    monkeypatch.setattr(stream_drain, "_drain_downstream", lambda nodes, extra_args: _fake_completed())

    result = run_stream(upstream="asr.transcribe", downstream=["asr.agreement"], sleep_fn=_no_sleep)
    assert result["upstream_returncode"] == 1


def test_run_stream_writes_log_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stream_drain, "_launch_upstream",
        lambda node, args: FakePopen(poll_sequence=[0], wait_returncode=0),
    )
    monkeypatch.setattr(stream_drain, "_drain_downstream", lambda nodes, extra_args: _fake_completed())

    log_path = tmp_path / "stream_test.log"
    run_stream(
        upstream="asr.transcribe", downstream=["asr.agreement"],
        sleep_fn=_no_sleep, log_path=log_path,
    )
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "asr.transcribe" in content
    assert "final drain" in content.lower()


# ---------------------------------------------------------------------------
# _drain_downstream() -- command construction (real function, not mocked)
# ---------------------------------------------------------------------------

def test_drain_downstream_solo_builds_pipe_run_command(monkeypatch):
    captured = {}

    def fake_run(cmd, cwd=None, capture_output=None, text=None):
        captured["cmd"] = cmd
        return _fake_completed()

    monkeypatch.setattr(stream_drain.subprocess, "run", fake_run)
    stream_drain._drain_downstream(["asr.agreement"], {})
    assert captured["cmd"][-2:] == ["run", "asr.agreement"]


def test_drain_downstream_multi_builds_run_many_command(monkeypatch):
    captured = {}

    def fake_run(cmd, cwd=None, capture_output=None, text=None):
        captured["cmd"] = cmd
        return _fake_completed()

    monkeypatch.setattr(stream_drain.subprocess, "run", fake_run)
    stream_drain._drain_downstream(["asr.agreement", "g2p"], {})
    cmd = captured["cmd"]
    assert cmd[-4:] == ["run-many", "asr.agreement", "--", "g2p"]


def test_drain_downstream_threads_per_node_extra_args(monkeypatch):
    captured = {}

    def fake_run(cmd, cwd=None, capture_output=None, text=None):
        captured["cmd"] = cmd
        return _fake_completed()

    monkeypatch.setattr(stream_drain.subprocess, "run", fake_run)
    stream_drain._drain_downstream(["filter.acoustic"], {"filter.acoustic": ["--workers", "8"]})
    assert captured["cmd"][-3:] == ["filter.acoustic", "--workers", "8"]
