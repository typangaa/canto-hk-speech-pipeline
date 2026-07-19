import types

import pytest

from pipeline.tools import chain_runner
from pipeline.tools.chain_runner import build_rounds, run_chain, _parse_round_set


def _fake_result(returncode=0):
    return types.SimpleNamespace(returncode=returncode)


# ---------------------------------------------------------------------------
# build_rounds() -- pure structure
# ---------------------------------------------------------------------------

def test_build_rounds_order_and_numbering():
    rounds = build_rounds(devices=None)
    assert [r.number for r in rounds] == list(range(1, 13))


def test_build_rounds_expected_run_many_pairs():
    rounds = {r.number: r for r in build_rounds(devices=None)}
    assert rounds[2].nodes == ["ingest.probe", "lang_screen.auto"]
    assert rounds[5].nodes == ["pregate.snr", "label.suite"]
    assert rounds[11].nodes == ["g2p", "tier.assign", "speaker.cluster"]
    assert rounds[2].is_run_many
    assert rounds[5].is_run_many
    assert rounds[11].is_run_many


def test_build_rounds_solo_rounds_are_single_node():
    rounds = build_rounds(devices=None)
    solo_numbers = {1, 3, 4, 6, 7, 8, 9, 10, 12}
    for r in rounds:
        if r.number in solo_numbers:
            assert len(r.nodes) == 1
            assert not r.is_run_many


def test_build_rounds_devices_threaded_only_to_gpu_rounds():
    rounds = {r.number: r for r in build_rounds(devices="cuda:0,cuda:1")}
    assert rounds[3].extra_args["segment.diarize"] == ["--devices", "cuda:0,cuda:1"]
    assert rounds[5].extra_args["label.suite"] == ["--devices", "cuda:0,cuda:1"]
    assert rounds[6].extra_args["asr.transcribe"] == ["--devices", "cuda:0,cuda:1"]
    assert rounds[2].extra_args["lang_screen.auto"] == ["--devices", "cuda:0,cuda:1"]
    # non-GPU rounds get no extra_args at all
    assert rounds[1].extra_args == {}
    assert rounds[11].extra_args == {}
    # pregate.snr is CPU-only -- must not get --devices even though it shares
    # round 5 with the GPU-only label.suite
    assert "pregate.snr" not in rounds[5].extra_args


def test_build_rounds_no_devices_means_no_extra_args():
    rounds = build_rounds(devices=None)
    assert all(r.extra_args == {} for r in rounds)


# ---------------------------------------------------------------------------
# _parse_round_set()
# ---------------------------------------------------------------------------

def test_parse_round_set_none():
    assert _parse_round_set(None) is None


def test_parse_round_set_parses_csv():
    assert _parse_round_set("2,11") == {2, 11}


def test_parse_round_set_strips_whitespace():
    assert _parse_round_set(" 2 , 11 ") == {2, 11}


# ---------------------------------------------------------------------------
# run_chain() -- dry-run never touches subprocess
# ---------------------------------------------------------------------------

def test_run_chain_dry_run_never_calls_subprocess(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called in --dry-run")

    monkeypatch.setattr(chain_runner.subprocess, "run", _boom)
    result = run_chain(dry_run=True)
    assert len(result["rounds"]) == 12
    assert all(r["dry_run"] for r in result["rounds"])


# ---------------------------------------------------------------------------
# run_chain() -- only / skip filtering
# ---------------------------------------------------------------------------

def test_run_chain_only_filters_to_requested_rounds(monkeypatch):
    monkeypatch.setattr(chain_runner.subprocess, "run", lambda *a, **k: _fake_result())
    result = run_chain(only={2, 11}, dry_run=False)
    assert [r["round"] for r in result["rounds"]] == [2, 11]


def test_run_chain_skip_excludes_requested_rounds(monkeypatch):
    monkeypatch.setattr(chain_runner.subprocess, "run", lambda *a, **k: _fake_result())
    result = run_chain(skip={1, 3, 4, 5, 6}, dry_run=False)
    assert [r["round"] for r in result["rounds"]] == [2, 7, 8, 9, 10, 11, 12]


# ---------------------------------------------------------------------------
# run_chain() -- command construction
# ---------------------------------------------------------------------------

def test_run_chain_solo_round_invokes_pipe_run(monkeypatch):
    calls = []

    def _record(cmd, cwd=None):
        calls.append(cmd)
        return _fake_result()

    monkeypatch.setattr(chain_runner.subprocess, "run", _record)
    run_chain(only={1}, dry_run=False)
    assert len(calls) == 1
    assert calls[0][-3:] == ["run", "run", "ingest.commit"] or "run" in calls[0]
    assert calls[0][-2:] == ["run", "ingest.commit"]


def test_run_chain_run_many_round_uses_dashdash_separators(monkeypatch):
    calls = []

    def _record(cmd, cwd=None):
        calls.append(cmd)
        return _fake_result()

    monkeypatch.setattr(chain_runner.subprocess, "run", _record)
    run_chain(only={11}, dry_run=False)
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[-6:] == ["run-many", "g2p", "--", "tier.assign", "--", "speaker.cluster"]


def test_run_chain_run_many_round_2_pairs_probe_and_lang_screen(monkeypatch):
    calls = []

    def _record(cmd, cwd=None):
        calls.append(cmd)
        return _fake_result()

    monkeypatch.setattr(chain_runner.subprocess, "run", _record)
    run_chain(only={2}, dry_run=False)
    cmd = calls[0]
    assert "ingest.probe" in cmd
    assert "lang_screen.auto" in cmd
    assert cmd.index("--") > cmd.index("ingest.probe")


def test_run_chain_run_many_round_5_pairs_pregate_and_label_suite(monkeypatch):
    calls = []

    def _record(cmd, cwd=None):
        calls.append(cmd)
        return _fake_result()

    monkeypatch.setattr(chain_runner.subprocess, "run", _record)
    run_chain(only={5}, dry_run=False)
    cmd = calls[0]
    assert "pregate.snr" in cmd
    assert "label.suite" in cmd
    assert cmd.index("--") > cmd.index("pregate.snr")


def test_run_chain_run_many_round_5_threads_devices_to_label_suite_only(monkeypatch):
    calls = []

    def _record(cmd, cwd=None):
        calls.append(cmd)
        return _fake_result()

    monkeypatch.setattr(chain_runner.subprocess, "run", _record)
    run_chain(only={5}, devices="cuda:0,cuda:1", dry_run=False)
    cmd = calls[0]
    devices_idx = cmd.index("label.suite") + 1
    assert cmd[devices_idx:devices_idx + 2] == ["--devices", "cuda:0,cuda:1"]
    pregate_idx = cmd.index("pregate.snr")
    assert cmd[pregate_idx + 1] in ("--", )


# ---------------------------------------------------------------------------
# run_chain() -- failure short-circuit
# ---------------------------------------------------------------------------

def test_run_chain_stops_on_first_failed_round(monkeypatch):
    calls = []

    def _record(cmd, cwd=None):
        calls.append(cmd)
        # fail round 8 (filter.text)
        if "filter.text" in cmd:
            return _fake_result(returncode=1)
        return _fake_result()

    monkeypatch.setattr(chain_runner.subprocess, "run", _record)
    result = run_chain(only={7, 8, 9, 10}, dry_run=False)
    executed_rounds = [r["round"] for r in result["rounds"]]
    assert executed_rounds == [7, 8]  # 9 and 10 never ran
    assert result["rounds"][-1]["returncode"] == 1


def test_run_chain_all_success_returns_zero_returncodes(monkeypatch):
    monkeypatch.setattr(chain_runner.subprocess, "run", lambda *a, **k: _fake_result())
    result = run_chain(only={7, 8}, dry_run=False)
    assert all(r["returncode"] == 0 for r in result["rounds"])


# ---------------------------------------------------------------------------
# run_chain() -- log file
# ---------------------------------------------------------------------------

def test_run_chain_writes_log_file(tmp_path, monkeypatch):
    monkeypatch.setattr(chain_runner.subprocess, "run", lambda *a, **k: _fake_result())
    log_path = tmp_path / "chain_test.log"
    run_chain(only={1}, dry_run=False, log_path=log_path)
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "Round 1" in content


def test_run_chain_dry_run_does_not_write_log_file(tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("must not call subprocess in dry-run")
    monkeypatch.setattr(chain_runner.subprocess, "run", _boom)
    log_path = tmp_path / "chain_test_dry.log"
    run_chain(dry_run=True, log_path=log_path)
    assert not log_path.exists()
