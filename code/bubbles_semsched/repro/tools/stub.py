#!/usr/bin/env python3
"""Stand-in for the Python interpreter: records a call (script, arguments, exported environment that differs from the baseline) in
$STUB_LOG and creates the named output file. Used to replay chain scripts WITHOUT running any simulation."""
import sys, os, json
base = json.load(open(os.environ['STUB_BASE']))
skip = {'PWD', 'OLDPWD', 'SHLVL', '_', 'HOME', 'STUB', 'STUB_BASE', 'STUB_LOG', 'AS_RUN', 'PY', 'SBX', 'BASHOPTS', 'SHELLOPTS', 'COLUMNS', 'LINES'}
env = {k: v for k, v in os.environ.items() if k not in skip and base.get(k) != v}
a = [os.path.basename(sys.argv[1])] + sys.argv[2:]
with open(os.environ['STUB_LOG'], 'a') as f:
    f.write(json.dumps(dict(run=os.environ.get('AS_RUN', '?'), argv=a, env=env)) + '\n')
for x in a[1:3]:
    if x.endswith('.json') and not os.path.exists(x):
        open(x, 'w').write('{"stub": true}')
