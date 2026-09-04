#!/usr/bin/env python3
"""Report fixed-anchor resident-kernel invariants from a saved checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaxrl.checkpoint import (
    CheckpointManager,
    fixed_anchor_norm_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        help="Checkpoint directory, or a run directory containing recovery checkpoints.",
    )
    parser.add_argument("--expected-members", type=int, default=8)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not (checkpoint / "critic.msgpack").is_file():
        checkpoint = CheckpointManager.resolve_recovery_checkpoint(checkpoint)
    report = fixed_anchor_norm_report(checkpoint)
    if report is None:
        raise SystemExit(f"checkpoint has no fixed anchors: {checkpoint}")
    if report["member_count"] != args.expected_members:
        raise SystemExit(
            "unexpected fixed-anchor member count: "
            f"{report['member_count']} != {args.expected_members}"
        )
    print(json.dumps({"checkpoint": str(checkpoint), **report}, indent=2))


if __name__ == "__main__":
    main()
