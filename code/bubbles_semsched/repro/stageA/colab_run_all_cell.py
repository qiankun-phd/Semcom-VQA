"""The whole GPU job as ONE background process on the Colab VM (the VM is kept alive by the Mac-side ping), two modes.

  drive mode (default): Drive must be mounted. Whatever is running in local mode is stopped, the local store is merged
      into the Drive folder, then the chain runs with Drive as the store. Merge rule, per detector run: the copy that is
      further along wins (finished > higher epoch; a tie keeps Drive); a superseded Drive copy is moved to
      superseded_<time>/, never deleted. Result files are merged only if both detectors came from the local store
      (otherwise they were computed with detectors that are not the ones kept).
  local mode (BUB_ROOT=/content/local_store/BUBBLES_hoverreport): Drive not mounted yet. Same chain, same folder layout,
      store on the VM disk - lost if the VM is lost; it exists so the GPU is not idle while waiting for the mount.
      A detector run that is already in progress is left to finish first.

Chain: detector training (colab_run_training_cell.py: G then O) -> Stage A (colab_run_stageA_cell.py). Every step skips
itself when its output is already in the store, so the job can be restarted at any time, in either mode.
Start it with colab_launch_all.py (drive mode) or colab_launch_local.py (local mode)."""
import os, subprocess, shutil, time, glob, signal

W = os.environ.get('BUB_DATA', '/content')   # working directory of the job (Colab VM: /content; a server: e.g. ~/bub_work)

LS, DRV = '/content/local_store/BUBBLES_hoverreport', '/content/drive/MyDrive/BUBBLES_hoverreport'
ROOT = os.environ.get('BUB_ROOT', DRV)
LOCAL = ROOT != DRV
NEED = ['colab_train_det.py', 'colab_run_training_cell.py', 'colab_run_stageA_cell.py', 'colab_setup_env.py', 'exp_scripts.tar.gz']
STEPS = '[c]olab_train_det.py|[c]olab_setup_env.py|[e]val_gated_sweep.py|[s]tageA_|[c]olab_preflight'
missing = [f for f in NEED if not os.path.exists(f'{W}/{f}')]
assert not missing, f'not uploaded: {missing}'
assert os.path.exists(f'{W}/visdrone_yolo/visdrone.yaml'), 'dataset not staged'
assert not os.path.exists(os.path.expanduser('~/phd_research/BUBBLES_equalbyte_20260929/STANDIN_DETECTORS')), 'stand-in detectors still present'


def pids(pat):
    out = subprocess.run(f'pgrep -f "{pat}"', shell=True, capture_output=True, text=True).stdout.split()
    return [p for p in out if int(p) not in (os.getpid(), os.getppid())]


def note(msg):
    line = f'[all:{"local" if LOCAL else "drive"}] {msg} {time.strftime("%H:%M:%S", time.gmtime())}'
    print(line, flush=True)
    with open(f'{W}/hb.log', 'a') as f:
        f.write(line + '\n')


def stop(pat, what):
    """signal the matching processes by pid (never this process: the pattern may match it too)"""
    if not pids(pat):
        return
    note(f'stopping {what} (pids {pids(pat)})')
    for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 5)):
        for p in pids(pat):
            try:
                os.kill(int(p), sig)
            except ProcessLookupError:
                pass
        for _ in range(wait):
            if not pids(pat):
                return
            time.sleep(1)


def put(src, dst):
    shutil.copyfile(src, dst + '.tmp'); os.replace(dst + '.tmp', dst)


def progress(run_dir):
    """how far a detector run is: 10**6 finished, 10**5 trained but not finalised, else epoch index of the newest
    readable checkpoint, -1 nothing usable"""
    if os.path.exists(f'{run_dir}/DONE.json'):
        return 10 ** 6
    import torch
    for cand in ('last.pt', 'last_prev.pt'):
        if os.path.exists(f'{run_dir}/{cand}'):
            try:
                ep = torch.load(f'{run_dir}/{cand}', map_location='cpu', weights_only=False).get('epoch')   # our own files
                return 10 ** 5 if ep is None or ep < 0 else int(ep)
            except Exception as e:
                note(f'{run_dir}/{cand} unreadable ({type(e).__name__})')
    return -1


def merge():
    for sub in ('runs', 'weights', 'results', 'scripts'):
        os.makedirs(f'{DRV}/{sub}', exist_ok=True)
    names = sorted(os.listdir(f'{LS}/runs')) if os.path.isdir(f'{LS}/runs') else []
    local_won = []
    for name in names:
        src, dst = f'{LS}/runs/{name}', f'{DRV}/runs/{name}'
        pl, pd = progress(src), progress(dst)
        if pl <= pd:
            note(f'merge {name}: Drive copy kept (progress Drive {pd}, local {pl})')
            continue
        old = [f for f in (os.listdir(dst) if os.path.isdir(dst) else []) if os.path.isfile(f'{dst}/{f}')]
        if old:
            keep = f'{dst}/superseded_{time.strftime("%Y%m%d_%H%M%S", time.gmtime())}'
            os.makedirs(keep)
            for f in old:
                os.replace(f'{dst}/{f}', f'{keep}/{f}')
        os.makedirs(dst, exist_ok=True)
        if os.path.exists(f'{LS}/weights/{name}_best.pt'):
            put(f'{LS}/weights/{name}_best.pt', f'{DRV}/weights/{name}_best.pt')
        files = sorted((f for f in os.listdir(src) if os.path.isfile(f'{src}/{f}') and not f.endswith('.tmp')), key=lambda f: f == 'DONE.json')
        for f in files:   # DONE.json last
            put(f'{src}/{f}', f'{dst}/{f}')
        local_won.append(name)
        note(f'merge {name}: local copy moved to Drive (progress local {pl}, Drive {pd}; {len(old)} old Drive files kept in superseded_*)')
    res = [f for f in glob.glob(f'{LS}/results/*') if os.path.isfile(f)]
    if res and set(local_won) >= {'v8m_visdrone', 'v8n_visdrone'}:
        new = [f for f in res if not os.path.exists(f'{DRV}/results/{os.path.basename(f)}')]
        for f in new:
            put(f, f'{DRV}/results/{os.path.basename(f)}')
        note(f'merge results: {len(new)} file(s) copied to Drive')
    elif res:
        note(f'merge results: {len(res)} local result file(s) NOT copied - their detectors are not the ones kept on Drive')
    if os.path.isdir(LS):
        os.rename(LS, f'{LS}_merged_{int(time.time())}')


others = pids('[c]olab_run_all_cell.py')
if LOCAL:
    assert not others, f'another job is already running (pids {others})'
    assert not os.path.ismount('/content/drive'), 'Drive is mounted: use drive mode'
    for sub in ('runs', 'weights', 'results', 'scripts'):
        os.makedirs(f'{ROOT}/{sub}', exist_ok=True)
    note('started; waiting for a detector run in progress' if pids('[c]olab_train_det.py') else 'started')
    while pids('[c]olab_train_det.py'):
        time.sleep(5)
else:
    assert os.path.ismount('/content/drive') and os.path.isdir(DRV), 'Drive is not mounted'
    note('started')
    stop('[c]olab_run_all_cell.py', 'the local-mode job')
    stop(STEPS, 'its running step')
    stop('envs/ff/bin/[f]fmpeg', 'encoder processes')
    merge()

os.environ['BUB_ROOT'] = ROOT
for f in (f'{W}/colab_run_training_cell.py', f'{W}/colab_run_stageA_cell.py'):
    exec(compile(open(f).read(), f, 'exec'), {'__name__': '__main__'})
note('JOB FINISHED')
