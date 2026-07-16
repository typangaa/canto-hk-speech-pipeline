"""
tests/test_prune_logs.py
T12: pipeline/tools/prune_logs.py -- gzip-then-delete log retention.
"""
import gzip
import os
import time

import pytest

from pipeline.tools.prune_logs import prune_logs


def _touch(path, age_days):
    path.write_text("log content\n" * 10)
    old_time = time.time() - age_days * 86400
    os.utime(path, (old_time, old_time))


def test_prune_logs_empty_dir_noop(tmp_path):
    result = prune_logs(logs_dir=tmp_path / "missing")
    assert result == {"gzipped": [], "deleted": [], "bytes_reclaimed": 0}


def test_prune_logs_leaves_recent_log_untouched(tmp_path):
    recent = tmp_path / "recent.log"
    _touch(recent, age_days=1)

    result = prune_logs(logs_dir=tmp_path, gzip_after_days=7, delete_after_days=60)

    assert result["gzipped"] == []
    assert recent.exists()


def test_prune_logs_gzips_old_log_and_removes_original(tmp_path):
    old = tmp_path / "old.log"
    _touch(old, age_days=10)

    result = prune_logs(logs_dir=tmp_path, gzip_after_days=7, delete_after_days=60)

    assert result["gzipped"] == [str(old)]
    assert not old.exists()
    gz_path = tmp_path / "old.log.gz"
    assert gz_path.exists()
    with gzip.open(gz_path, "rt") as f:
        assert f.read() == "log content\n" * 10


def test_prune_logs_deletes_old_gz_archive(tmp_path):
    gz_path = tmp_path / "ancient.log.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(b"stale")
    old_time = time.time() - 90 * 86400
    os.utime(gz_path, (old_time, old_time))

    result = prune_logs(logs_dir=tmp_path, gzip_after_days=7, delete_after_days=60)

    assert result["deleted"] == [str(gz_path)]
    assert not gz_path.exists()


def test_prune_logs_dry_run_makes_no_changes(tmp_path):
    old = tmp_path / "old.log"
    _touch(old, age_days=10)
    gz_path = tmp_path / "ancient.log.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(b"stale")
    os.utime(gz_path, (time.time() - 90 * 86400,) * 2)

    result = prune_logs(logs_dir=tmp_path, gzip_after_days=7, delete_after_days=60, dry_run=True)

    assert result["gzipped"] == [str(old)]
    assert result["deleted"] == [str(gz_path)]
    assert old.exists()
    assert gz_path.exists()
    assert result["bytes_reclaimed"] == 0


def test_prune_logs_skips_already_gzipped_recent_archive(tmp_path):
    gz_path = tmp_path / "recent.log.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(b"fresh")
    os.utime(gz_path, (time.time() - 1 * 86400,) * 2)

    result = prune_logs(logs_dir=tmp_path, gzip_after_days=7, delete_after_days=60)

    assert result["deleted"] == []
    assert gz_path.exists()


def test_prune_logs_reclaims_bytes_on_gzip(tmp_path):
    old = tmp_path / "old.log"
    _touch(old, age_days=10)

    result = prune_logs(logs_dir=tmp_path, gzip_after_days=7, delete_after_days=60)

    assert result["bytes_reclaimed"] > 0
