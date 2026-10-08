"""Preflight on a VM without Drive: build the environment and smoke-test every Stage-A script on a few images with
STAND-IN detectors (official yolov8n.pt). Purpose: find environment errors (Python 3.13 / torch 2.11 / numpy 2 / timm /
ffmpeg) now, not hours into the unattended run. Nothing computed here is a result: all outputs and the stand-in
detector files are deleted at the end. Log: /content/preflight.log; one line per step in /content/hb.log.
Run in the background:  nohup python3 /content/colab_preflight.py > /content/preflight.log 2>&1 &"""
import subprocess, time, os, glob, shutil

EQ = '/root/phd_research/BUBBLES_equalbyte_20260929'
HR = '/root/phd_research/BUBBLES_hoverreport_20260930'
R = f'{HR}/results'
SWEEP = ['--masks', 'det', 'oracle', '--snrs', '-2', '1', '4', '7', '10', '13', '--conds', 'rtk', 'gnss', '--changes', '0.1', '--limit', '4']
STEPS = [
    ('setup', ['python3', '/content/colab_setup_env.py']),
    ('sweep_c96', ['python3', f'{HR}/scripts/eval_gated_sweep.py', '--C', '96'] + SWEEP + ['--out', f'{R}/pf_sweep_c96.json']),
    ('sweep_c192', ['python3', f'{HR}/scripts/eval_gated_sweep.py', '--C', '192'] + SWEEP + ['--out', f'{R}/pf_sweep_c192.json']),
    ('hevc_base', ['python3', f'{HR}/scripts/stageA_hevc.py', 'base', '2', '4']),
    ('hevc_reg', ['python3', f'{HR}/scripts/stageA_hevc.py', 'reg', '2', '4']),
    ('sym', ['python3', f'{HR}/scripts/stageA_sym.py']),
]


def note(msg):
    line = f'[preflight] {msg} {time.strftime("%H:%M:%S", time.gmtime())}'
    print(line, flush=True)
    with open('/content/hb.log', 'a') as f:
        f.write(line + '\n')


env = dict(os.environ, PREFLIGHT='1')
codes = {}
try:
    for name, cmd in STEPS:
        note(f'{name}: start')
        t0 = time.time()
        rc = subprocess.run(cmd, env=env, cwd=HR if os.path.isdir(HR) else '/content').returncode
        codes[name] = rc
        note(f'{name}: exit code {rc} after {time.time() - t0:.0f}s')
        if name == 'setup' and rc != 0:
            break
finally:
    if os.path.exists(f'{R}/colab_env_report.json'):
        shutil.copy(f'{R}/colab_env_report.json', '/content/preflight_env_report.json')
    for f in glob.glob(f'{R}/*'):                       # nothing from a stand-in run may be mistaken for a result
        os.remove(f)
    for f in (f'{EQ}/runs/v8m_visdrone/weights/best.pt', f'{EQ}/weights/onboard_v8n_visdrone.pt', f'{EQ}/STANDIN_DETECTORS'):
        if os.path.exists(f):
            os.remove(f)
    note(f'PREFLIGHT DONE codes={codes} (stand-in detectors and outputs removed)')
