from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit a pinned Behavior-to-Weights AWS Batch job"
    )
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--job-queue", required=True)
    parser.add_argument("--job-definition", required=True)
    parser.add_argument("--command", nargs="+", required=True)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    request = {
        "jobName": args.job_name,
        "jobQueue": args.job_queue,
        "jobDefinition": args.job_definition,
        "containerOverrides": {"command": args.command},
        "parameters": json.loads(args.parameters.read_text()) if args.parameters else {},
        "tags": {"project": "behavior-to-weights"},
        "propagateTags": True,
    }
    if args.dry_run:
        print(json.dumps(request, indent=2))
        return
    try:
        import boto3
    except ImportError as error:
        raise SystemExit("Install behavior2weights[cloud] before submitting") from error
    response = boto3.client("batch").submit_job(**request)
    print(json.dumps({"jobId": response["jobId"], "jobName": response["jobName"]}, indent=2))


if __name__ == "__main__":
    main()
