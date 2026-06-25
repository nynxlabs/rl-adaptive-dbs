#!/usr/bin/env python3
"""Check Mehregan DDPG replication summary against the §IV checklist.

Example::

    uv run python scripts/check_mehregan_replication.py artifacts/ddpg/paper_train0_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from controllers.ddpg.checklist import assess_replication_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assess Mehregan replication summary JSON")
    parser.add_argument("summary", type=Path, help="Replication summary JSON path")
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args(argv)

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    report = assess_replication_summary(payload)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["passed"] else "FAIL"
        print(f"{status} variant={report['variant']}")
        for check in report["checks"]:
            mark = "ok" if check["passed"] else "FAIL"
            print(f"  [{mark}] {check['name']}: {check['detail']}")
        print(report["paper_notes"])

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
