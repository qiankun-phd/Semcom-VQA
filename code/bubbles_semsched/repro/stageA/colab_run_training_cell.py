"""Run the detector training chain INSIDE the kernel, printing a heartbeat line every 30 s (kernel output).
G (yolov8m, v8m_visdrone) then O (yolov8n, v8n_visdrone); each resumes from its Drive checkpoint if one exists.
Training output goes to /content/train_main.log, heartbeats also to /content/hb.log (read both with `colab download`).
Submit with a long client timeout:  colab exec -s bub --timeout 14400 -f colab_run_training_cell.py
Refuses to start if a trainer is already running, so a retried submission cannot start a second one.

History: v1 of this cell was silent (subprocess.run, no output) and the VM was still reclaimed 11-14 min after it
started, so 'a busy kernel is enough' was wrong. The heartbeat + attached client is the second attempt."""
import subprocess, time, os, sys
W = os.environ.get('BUB_DATA', '/content')   # working directory of the job (Colab VM: /content; a server: e.g. ~/bub_work)

T0 = time.time(); EVERY = 30
running = subprocess.run('pgrep -f "[c]olab_train_det.py"', shell=True, capture_output=True, text=True).stdout.split()
assert not running, f'a trainer is already running (pids {running}) - not starting another'
assert not os.environ.get('BUB_ROOT', '/content/drive/').startswith('/content/drive/') or os.path.ismount('/content/drive'), 'Drive is not mounted'
assert os.path.exists(f'{W}/visdrone_yolo/visdrone.yaml'), 'dataset not staged'


def hb(msg):
    line = f'[hb] {time.strftime("%H:%M:%S", time.gmtime())} +{(time.time() - T0) / 60:.1f}min {msg}'
    print(line, flush=True)
    with open(f'{W}/hb.log', 'a') as f:
        f.write(line + '\n')


def last_sync():
    try:
        lines = [l for l in open(f'{W}/train_main.log', errors='replace').read().splitlines() if l.startswith('[sync]')]
        return lines[-1][:60] if lines else 'no epoch synced yet in this session'
    except OSError:
        return 'no log yet'


with open(f'{W}/train_main.log', 'a') as log:
    log.write(f'[chain] started {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}\n'); log.flush()
    for model, name in (('yolov8m.pt', 'v8m_visdrone'), ('yolov8n.pt', 'v8n_visdrone')):
        p = subprocess.Popen([sys.executable, f'{W}/colab_train_det.py', '--model', model, '--name', name, '--epochs', '30'],
                             stdout=log, stderr=subprocess.STDOUT)
        hb(f'{name}: started'); last = time.time()
        while p.poll() is None:
            time.sleep(1)
            if time.time() - last >= EVERY:
                hb(f'{name}: {last_sync()}'); last = time.time()
        log.write(f'[chain] {name} exit code {p.returncode} at {time.strftime("%H:%M:%S UTC", time.gmtime())}\n'); log.flush()
        hb(f'{name}: exit code {p.returncode}')
        if p.returncode != 0:
            break
    log.write('[chain] finished\n'); log.flush()
hb('training chain finished')
