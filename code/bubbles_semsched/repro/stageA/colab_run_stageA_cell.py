"""Stage A, run INSIDE the kernel (blocking): environment setup, a small smoke pass, then the experiments whose results
the capacity model and the scheduling simulator read. Needs both detectors finished (Drive weights/ + DONE.json).

  setup       colab_setup_env.py
  smoke       every script on 4 images (results named *_smoke.json, not used for anything)
  sym         stageA_sym.py                      -> sym_baseline_synth.json
  sweep_c192  eval_gated_sweep.py --C 192 ...    -> eval_gated_sweep_c192.json     (original arguments)
  sweep_c96   eval_gated_sweep.py --C 96 ...     -> eval_gated_sweep_c96.json
  hevc_base   stageA_hevc.py base                -> ref_gain_sub.json
  hevc_reg    stageA_hevc.py reg                 -> ref_gain_sub_reg.json

After every step all result JSONs and the log are copied to Drive results/. A step whose output is already on Drive is
skipped (the file is copied back instead), so the cell can simply be re-submitted on a new VM.
A heartbeat line is printed every 30 s (kernel output) and appended to /content/hb.log.
Log: /content/stageA.log (read it with `colab download`). Submit: colab exec -s bub --timeout 14400 -f colab_run_stageA_cell.py"""
import subprocess, time, os, glob, shutil, sys
W = os.environ.get('BUB_DATA', '/content')   # working directory of the job (Colab VM: /content; a server: e.g. ~/bub_work)
NW = os.environ.get('BUB_HEVC_WORKERS', '2')   # encoder processes for the HEVC steps (Colab VM: 2 vCPUs; 182 used 12)

DR = os.environ.get('BUB_ROOT', '/content/drive/MyDrive/BUBBLES_hoverreport')   # the store: Drive, or the VM disk before the mount
HR = os.path.expanduser('~/phd_research/BUBBLES_hoverreport_20260930')
LOG = f'{W}/stageA.log'
PY = sys.executable
busy = subprocess.run('pgrep -f "[c]olab_train_det.py|[c]olab_setup_env.py|[e]val_gated_sweep.py|[s]tageA_"', shell=True,
                      capture_output=True, text=True).stdout.split()
assert not busy, f'training or a stage-A step is still running (pids {busy})'
assert not DR.startswith('/content/drive/') or os.path.ismount('/content/drive'), 'Drive is not mounted'
os.makedirs(f'{DR}/results', exist_ok=True)
SWEEP = ['--masks', 'det', 'oracle', '--snrs', '-2', '1', '4', '7', '10', '13', '--conds', 'rtk', 'gnss', '--changes', '0.1']
sweep = lambda C, extra=(): [PY, f'{HR}/scripts/eval_gated_sweep.py', '--C', str(C)] + SWEEP + list(extra)
R = f'{HR}/results'
STEPS = [  # name, output files (all must exist to skip), command, abort the chain on failure
    ('setup', [], [PY, f'{W}/colab_setup_env.py'], True),
    ('smoke_sweep', [], sweep(96, ['--limit', '4', '--out', f'{R}/eval_gated_sweep_c96_smoke.json']), False),
    ('smoke_hevc_base', [], [PY, f'{HR}/scripts/stageA_hevc.py', 'base', NW, '4'], False),
    ('smoke_hevc_reg', [], [PY, f'{HR}/scripts/stageA_hevc.py', 'reg', NW, '4'], False),
    ('sym', ['sym_baseline_synth.json'], [PY, f'{HR}/scripts/stageA_sym.py'], False),
    ('sweep_c192', ['eval_gated_sweep_c192.json'], sweep(192, ['--out', f'{R}/eval_gated_sweep_c192.json']), False),
    ('sweep_c96', ['eval_gated_sweep_c96.json'], sweep(96, ['--out', f'{R}/eval_gated_sweep_c96.json']), False),
    ('hevc_base', ['ref_gain_sub.json'], [PY, f'{HR}/scripts/stageA_hevc.py', 'base', NW], False),
    ('hevc_reg', ['ref_gain_sub_reg.json'], [PY, f'{HR}/scripts/stageA_hevc.py', 'reg', NW], False),
]


T0 = time.time()


def hb(msg):
    line = f'[hb] {time.strftime("%H:%M:%S", time.gmtime())} +{(time.time() - T0) / 60:.1f}min {msg}'
    print(line, flush=True)
    with open(f'{W}/hb.log', 'a') as f:
        f.write(line + '\n')


def sync():
    for f in glob.glob(f'{R}/*.json') + [LOG]:
        dst = f'{DR}/results/{os.path.basename(f)}'
        shutil.copyfile(f, dst + '.tmp'); os.replace(dst + '.tmp', dst)


with open(LOG, 'a') as log:
    say = lambda s: (log.write(f'[chainA] {s} | {time.strftime("%H:%M:%S UTC", time.gmtime())}\n'), log.flush())
    say('started')
    for name, outs, cmd, fatal in STEPS:
        if outs and all(os.path.exists(f'{DR}/results/{o}') for o in outs):
            os.makedirs(R, exist_ok=True)
            for o in outs:
                shutil.copyfile(f'{DR}/results/{o}', f'{R}/{o}')
            say(f'{name}: output already on Drive, skipped')
            continue
        say(f'{name}: start')
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HR if os.path.isdir(HR) else W)
        hb(f'{name}: started'); last = time.time()
        while p.poll() is None:
            time.sleep(1)
            if time.time() - last >= 30:
                hb(f'{name}: running {time.time() - t0:.0f}s'); last = time.time()
        rc = p.returncode
        hb(f'{name}: exit code {rc}')
        say(f'{name}: exit code {rc} after {time.time() - t0:.0f}s')
        if os.path.isdir(R):
            sync()
        if rc != 0 and fatal:
            say('aborting: a required step failed')
            break
    say('finished')
    log.flush()
if os.path.isdir(R):
    sync()
hb('stage A chain finished')
