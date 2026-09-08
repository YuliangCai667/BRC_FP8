#!/usr/bin/env python3
"""Wait only for the assigned GPU, launch once, then become the training process."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--gpu',type=int,required=True);p.add_argument('--format',required=True)
p.add_argument('--run-id',required=True);p.add_argument('--smoke-report',type=Path,required=True)
p.add_argument('--status',type=Path,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
state={'format':a.format,'gpu':a.gpu,'gpu_uuid':uuid,'run_id':a.run_id,'worktree':str(root),'commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'launcher_pid':os.getpid()}
a.status.parent.mkdir(parents=True,exist_ok=True)
# Exclusive per-run status file is the duplicate launch guard.
with a.status.open('x') as f: json.dump(state,f,indent=2)
def report(status):
    state.update(status=status,time=datetime.datetime.now().astimezone().isoformat())
    a.status.write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
report('WAITING_GPU')
while True:
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if not any(line.split(',')[0].strip()==uuid for line in apps.splitlines()): break
    time.sleep(30)
smoke=json.loads(a.smoke_report.read_text())
if smoke['status']!='PASSED' or smoke['format']!=a.format:
    report('FAILED_SMOKE');raise SystemExit('Passed format-specific smoke required')
report('LAUNCHING')
os.environ['BRC_RUN_ID']=a.run_id
os.chdir(root)
os.execv('/bin/bash',['bash',str(root/'scripts/run_v1_block_compute.sh'),str(a.gpu),a.format,'42','500000'])
