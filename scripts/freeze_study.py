from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from behavior2weights.compute.runtime import runtime_info_dict
from behavior2weights.utils import file_sha256, git_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a tamper-evident preregistration lock file"
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = []
    for path in sorted(args.files):
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(
            {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
        )
    payload = {
        "schema_version": 1,
        "study_id": args.study_id,
        "frozen_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "files": entries,
        "git": git_metadata(),
        "runtime": runtime_info_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
