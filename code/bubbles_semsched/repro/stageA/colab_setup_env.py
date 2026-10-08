#!/usr/bin/env python3
"""Build, on the Colab VM, the directory layout the experiment scripts expect (same paths as on server 182, so the
scripts run unmodified), and report exactly what was installed.

  /root/phd_research/BUBBLES_equalbyte_20260929/{scripts/eq_byte.py, runs/v8m_visdrone/weights/best.pt, weights/onboard_v8n_visdrone.pt}
  /root/phd_research/BUBBLES_hoverreport_20260930/{scripts, results, weights, SwinJSCC}
  /root/.local/share/mamba/envs/ff/bin/ffmpeg        (conda-forge ffmpeg 7.1.1, the version used on 182)

Inputs: /content/exp_scripts.tar.gz (uploaded), detector weights in Drive weights/ (written by colab_train_det.py).
SwinJSCC: official code at commit a6d0e6d; the two fixed-point models used by the gated experiments (SNR10, C96 / C192)
from the authors' Google Drive folder, cached in our Drive folder. They are third-party files: loaded with
weights_only=True only (tensors, no code execution).
Differences from 182 that this script cannot remove are written to the report: torch / timm / ultralytics / x265
versions, and the detectors themselves (retrained here).

PREFLIGHT=1 (environment variable): no Drive needed. Everything is built the same way except that the two detector
files are STAND-INS (official yolov8n.pt, COCO classes) so the experiment scripts can be smoke-tested before the real
detectors exist. A marker file STANDIN_DETECTORS is written next to them; the normal run overwrites both files from
Drive and removes the marker. Nothing computed with stand-ins is a result."""
import os, sys, json, shutil, hashlib, tarfile, subprocess, time

DR = os.environ.get('BUB_ROOT', '/content/drive/MyDrive/BUBBLES_hoverreport')   # the store: Drive, or the VM disk before the mount
W = os.environ.get('BUB_DATA', '/content')   # working directory of the job (Colab VM: /content; a server: e.g. ~/bub_work)
HOME = os.path.expanduser('~')              # Colab VM: /root
EQ = f'{HOME}/phd_research/BUBBLES_equalbyte_20260929'
HR = f'{HOME}/phd_research/BUBBLES_hoverreport_20260930'
FF = f'{HOME}/.local/share/mamba/envs/ff/bin/ffmpeg'
SJ_COMMIT = 'a6d0e6d'
SJ_WEIGHTS = {'SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C96.model': '1g4stFrseKBe7N_hXNUSdLO5SWeiUzmjr',
              'SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C192.model': '18MbARWltc1OKw4yBTyxUv2HlEIBg4kYA'}
PRE = os.environ.get('PREFLIGHT') == '1'
MARK = f'{EQ}/STANDIN_DETECTORS'
rep = {'started': time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()), 'preflight': PRE, 'store': DR}


def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f'{cmd}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}')
    return (r.stdout + r.stderr).strip()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


assert PRE or not DR.startswith('/content/drive/') or (os.path.ismount('/content/drive') and os.path.isdir(DR)), 'Drive working folder not mounted'
for d in [f'{EQ}/scripts', f'{EQ}/weights', f'{EQ}/runs/v8m_visdrone/weights', f'{HR}/scripts', f'{HR}/results', f'{HR}/weights'] \
        + ([] if PRE else [f'{DR}/weights/swinjscc', f'{DR}/scripts', f'{DR}/results']):
    os.makedirs(d, exist_ok=True)

# 1. scripts
with tarfile.open(f'{W}/exp_scripts.tar.gz') as t:
    t.extractall(f'{HR}/scripts', filter='data')
shutil.copy(f'{HR}/scripts/eq_byte.py', f'{EQ}/scripts/eq_byte.py')
if not PRE:
    shutil.copy(f'{W}/exp_scripts.tar.gz', f'{DR}/scripts/exp_scripts.tar.gz')
rep['scripts'] = sorted(os.listdir(f'{HR}/scripts'))
print('[setup] scripts:', len(rep['scripts']), flush=True)

# 2. detectors (our own, trained in this project)
DETS = (f'{EQ}/runs/v8m_visdrone/weights/best.pt', f'{EQ}/weights/onboard_v8n_visdrone.pt')
if PRE:
    from ultralytics import YOLO
    os.makedirs(f'{W}/standin', exist_ok=True)
    os.chdir(f'{W}/standin'); YOLO('yolov8n.pt'); os.chdir(W)   # official base weights, downloaded by ultralytics
    open(MARK, 'w').write('stand-in detectors (official yolov8n.pt): smoke tests only\n')
    for dst in DETS:
        shutil.copyfile(f'{W}/standin/yolov8n.pt', dst)
    rep['detectors'] = 'STAND-IN official yolov8n.pt (COCO) at both paths - preflight only'
    print('[setup] detectors: STAND-INS (preflight)', flush=True)
else:
    for src, dst in zip((f'{DR}/weights/v8m_visdrone_best.pt', f'{DR}/weights/v8n_visdrone_best.pt'), DETS):
        assert os.path.exists(src), f'missing detector on Drive: {src}'
        shutil.copyfile(src, dst)
    if os.path.exists(MARK):
        os.remove(MARK)
    rep['detectors'] = {n: json.load(open(f'{DR}/runs/{n}/DONE.json')) for n in ('v8m_visdrone', 'v8n_visdrone')}
    print('[setup] detectors:', {k: round(v['map50'], 4) for k, v in rep['detectors'].items()}, flush=True)

# 3. python packages
import torch, ultralytics
for pkg in ('timm', 'gdown'):
    try:
        __import__(pkg)
    except ImportError:
        sh(f'{sys.executable} -m pip install -q {pkg}')
import timm, gdown, cv2, numpy
rep['versions'] = dict(python=sys.version.split()[0], torch=torch.__version__, ultralytics=ultralytics.__version__, timm=timm.__version__,
                       gdown=gdown.__version__, opencv=cv2.__version__, numpy=numpy.__version__, gpu=torch.cuda.get_device_name(0),
                       cpus=os.cpu_count())
print('[setup] versions:', rep['versions'], flush=True)

# 4. SwinJSCC code, pinned
if not os.path.isdir(f'{HR}/SwinJSCC/.git'):
    sh(f'git clone -q https://github.com/semcomm/SwinJSCC.git {HR}/SwinJSCC')
sh(f'git -C {HR}/SwinJSCC checkout -q {SJ_COMMIT}')
head = sh(f'git -C {HR}/SwinJSCC rev-parse --short=7 HEAD')
assert head == SJ_COMMIT, f'SwinJSCC is at {head}, expected {SJ_COMMIT}'
rep['swinjscc_commit'] = sh(f'git -C {HR}/SwinJSCC log -1 --format="%h %ci"')
print('[setup] SwinJSCC', rep['swinjscc_commit'], flush=True)

# 5. SwinJSCC weights (Drive cache first, otherwise the authors' Google Drive)
rep['swinjscc_weights'] = {}
for name, fid in SJ_WEIGHTS.items():
    dst, cache = f'{HR}/weights/{name}', f'{DR}/weights/swinjscc/{name}'
    if os.path.exists(dst):
        src = 'already on this VM'
    elif not PRE and os.path.exists(cache):
        shutil.copyfile(cache, dst); src = 'our Drive cache'
    else:
        gdown.download(id=fid, output=dst, quiet=True); src = f'authors Drive id {fid}'
        assert os.path.exists(dst), f'download failed: {name}'
    sd = torch.load(dst, map_location='cpu', weights_only=True)
    assert isinstance(sd, dict) and all(torch.is_tensor(v) for v in sd.values()), 'not a plain state dict'
    rep['swinjscc_weights'][name] = dict(bytes=os.path.getsize(dst), sha256=sha(dst), source=src, tensors=len(sd),
                                         params=int(sum(v.numel() for v in sd.values())))
    if not PRE and not os.path.exists(cache):
        shutil.copyfile(dst, cache + '.tmp'); os.replace(cache + '.tmp', cache)
    del sd
    print('[setup] weight', name, rep['swinjscc_weights'][name]['bytes'], src, flush=True)

# 6. ffmpeg 7.1.1 (conda-forge), at the path the scripts use
if not os.path.exists(FF):
    os.makedirs(f'{W}/tools', exist_ok=True)
    sh(f'cd {W}/tools && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba')
    sh(f'MAMBA_ROOT_PREFIX={HOME}/.local/share/mamba {W}/tools/bin/micromamba create -y -q -n ff -c conda-forge ffmpeg=7.1.1')
assert os.path.exists(FF), 'ffmpeg env was not created'
rep['ffmpeg'] = sh(f'{FF} -version').splitlines()[0]
enc = sh(f'{FF} -hide_banner -encoders')
assert 'libx265' in enc, 'this ffmpeg has no libx265'
x = sh(f'{HOME}/.local/share/mamba/envs/ff/bin/x265 --version', check=False)
rep['x265'] = next((l for l in x.splitlines() if 'version' in l.lower()), x[:200])
rep['system_ffmpeg_not_used'] = sh('ffmpeg -version', check=False).splitlines()[:1]
print('[setup]', rep['ffmpeg'], '|', rep['x265'], flush=True)

# 7. checks: dataset link, E1a split fingerprint, one SwinJSCC forward pass
import glob, random
vis = f'{HOME}/phd_research/uav-vqa-semantic-rl/data/processed/visdrone_yolo'
names = [os.path.basename(p) for p in sorted(glob.glob(f'{vis}/images/val/*.jpg'))]
random.Random(0).shuffle(names)
rep['e1a_split_sha256'] = hashlib.sha256('\n'.join(names).encode()).hexdigest()
rep['n_val'] = len(names)
sys.path.insert(0, f'{HR}/scripts')
import jscc_lib as J
net = J.build(f'{HR}/weights/SwinJSCC_wo_SAandRA_AWGN_HRimage_snr10_psnr_C96.model', model='SwinJSCC_w/o_SAandRA', C='96', snrs='10')
x = torch.rand(1, 3, 384, 640).cuda()
with torch.no_grad():
    J._set_res(net, 384, 640)
    z = net.encoder(x, 10, 96, net.model)
    z = z[0] if isinstance(z, (tuple, list)) else z
rep['swinjscc_forward'] = dict(input=[384, 640], latent_shape=list(z.shape))
print('[setup] SwinJSCC forward ok, latent', list(z.shape), '| val images', len(names), '| split', rep['e1a_split_sha256'][:16], flush=True)

rep['finished'] = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())
json.dump(rep, open(f'{HR}/results/colab_env_report.json', 'w'), indent=1)
if not PRE:
    shutil.copy(f'{HR}/results/colab_env_report.json', f'{DR}/results/colab_env_report.json')
print('[setup] SETUP DONE' + (' (PREFLIGHT, stand-in detectors)' if PRE else ''), flush=True)
