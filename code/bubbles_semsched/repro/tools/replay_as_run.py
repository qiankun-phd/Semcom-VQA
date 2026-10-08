#!/usr/bin/env python3
"""Replays every as-run chain script (as_run/run_*.sh) against tools/stub.py in a sandbox and writes .tmp/calls.jsonl: the exact commands
(arguments + exported environment) each chain issued. The scripts are rewritten on the fly only where they would touch the machine:
interpreter -> stub, cd ~/... -> sandbox, waits for another chain's *.done file and sleeps removed; HOME points into the sandbox.
usage: python tools/replay_as_run.py"""
import os, re, json, subprocess, glob, shutil, collections
REPRO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); T = os.path.join(REPRO, '.tmp'); SBX = os.path.join(T, 'sbx'); LOG = os.path.join(T, 'calls.jsonl')
os.makedirs(T, exist_ok=True)
if os.path.exists(LOG):
    os.remove(LOG)
shutil.rmtree(SBX, ignore_errors=True); os.makedirs(SBX)
open(os.path.join(SBX, 'sim_inputs.json'), 'w').write('{}')
json.dump(dict(os.environ), open(os.path.join(T, 'base_env.json'), 'w'))
stub = f'python3 {REPRO}/tools/stub.py'
for f in sorted(glob.glob(os.path.join(REPRO, 'as_run', 'run_*.sh'))):
    s = open(f).read()
    s = re.sub(r'(?m)^(\s*)PY=.*$', r'\1PY="$STUB"', s)
    s = re.sub(r'(?m)^\s*while \[ ! -[fe] .*?done\s*$', 'true', s)
    s = re.sub(r'(?m)^(\s*)cd\s+(~|\$HOME)\S*', r'\1cd "$SBX"', s)
    s = re.sub(r'\bsleep\s+[0-9.]+[smh]?', 'true', s)
    s = re.sub(r'\bpython3?\b(?=\s+\S+\.py)', '$STUB', s)
    p = os.path.join(T, 'replay.sh'); open(p, 'w').write(s)
    env = dict(os.environ, HOME=SBX, SBX=SBX, STUB=stub, STUB_BASE=os.path.join(T, 'base_env.json'), STUB_LOG=LOG, AS_RUN=os.path.basename(f))
    subprocess.run(['bash', p], cwd=SBX, env=env, capture_output=True, text=True, timeout=180)
calls = [json.loads(l) for l in open(LOG)]
print(len(calls), 'commands from', len(collections.Counter(c['run'] for c in calls)), 'chain scripts ->', LOG)
