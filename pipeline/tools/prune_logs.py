"""
pipeline/tools/prune_logs.py
Log retention for metadata/logs/ -- gzip files older than --gzip-after-days, delete
gzipped archives older than --delete-after-days. Node FileHandlers write plain-text
`*.log`; ad-hoc shell-redirected batch logs (`t15_*.log` etc.) land here too -- both
are handled the same way since this operates on mtime, not on how the file was written.

Idempotent and safe to run repeatedly / on a schedule: already-gzipped files are
skipped by the gzip pass, and files newer than the threshold are left untouched.
"""
import argparse
import gzip
import shutil
import time
from pathlib import Path

from pipeline.config import LOGS_DIR

DEFAULT_GZIP_AFTER_DAYS = 7
DEFAULT_DELETE_AFTER_DAYS = 60


def _age_days(path: Path, now: float) -> float:
    return (now - path.stat().st_mtime) / 86400.0


def prune_logs(
    *,
    logs_dir: Path = LOGS_DIR,
    gzip_after_days: float = DEFAULT_GZIP_AFTER_DAYS,
    delete_after_days: float = DEFAULT_DELETE_AFTER_DAYS,
    dry_run: bool = False,
) -> dict:
    now = time.time()
    gzipped, deleted, bytes_reclaimed = [], [], 0

    if not logs_dir.exists():
        return {"gzipped": [], "deleted": [], "bytes_reclaimed": 0}

    for path in sorted(logs_dir.glob("*.log")):
        if _age_days(path, now) < gzip_after_days:
            continue
        target = path.with_suffix(path.suffix + ".gz")
        if dry_run:
            gzipped.append(str(path))
            continue
        with path.open("rb") as src, gzip.open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        bytes_reclaimed += path.stat().st_size - target.stat().st_size
        path.unlink()
        gzipped.append(str(path))

    for path in sorted(logs_dir.glob("*.log.gz")):
        if _age_days(path, now) < delete_after_days:
            continue
        if dry_run:
            deleted.append(str(path))
            continue
        bytes_reclaimed += path.stat().st_size
        path.unlink()
        deleted.append(str(path))

    return {"gzipped": gzipped, "deleted": deleted, "bytes_reclaimed": bytes_reclaimed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gzip-after-days", type=float, default=DEFAULT_GZIP_AFTER_DAYS)
    parser.add_argument("--delete-after-days", type=float, default=DEFAULT_DELETE_AFTER_DAYS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = prune_logs(
        gzip_after_days=args.gzip_after_days,
        delete_after_days=args.delete_after_days,
        dry_run=args.dry_run,
    )
    verb = "Would gzip" if args.dry_run else "Gzipped"
    print(f"{verb} {len(result['gzipped'])} file(s)")
    verb = "Would delete" if args.dry_run else "Deleted"
    print(f"{verb} {len(result['deleted'])} archive(s)")
    if not args.dry_run:
        print(f"Bytes reclaimed: {result['bytes_reclaimed']:,}")


if __name__ == "__main__":
    main()
