#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jaxrl.deployment_export import export_checkpoint

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-step', type=int, default=500000)
    a = p.parse_args()
    result = export_checkpoint(a.checkpoint, a.output, expected_step=a.expected_step)
    print(json.dumps(result['storage'], indent=2))
