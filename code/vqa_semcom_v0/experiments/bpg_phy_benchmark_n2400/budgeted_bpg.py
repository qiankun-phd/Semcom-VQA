"""Frozen first-fit libbpg wrapper. Does not change the production codec router."""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

SIDES = (384, 320, 256, 224, 192, 160, 128, 96, 64)
QPS = (20, 25, 30, 35, 40, 45, 50, 51)
NATIVE_MAGIC = b'BPG\xfb'

def resolve_binary(name):
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f'Missing executable: {name}; install libbpg before running')
    return str(Path(path).resolve())

def fingerprints(enc, dec):
    result={}
    for name,path in [('bpgenc',resolve_binary(enc)),('bpgdec',resolve_binary(dec))]:
        probe=subprocess.run([path,'-h'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
        first_line=(probe.stdout+probe.stderr).decode(errors='replace').splitlines()[0]
        if not first_line.endswith('version 0.9.8'):
            raise RuntimeError('Expected libbpg0.9.8: '+first_line)
        result[name]={'path':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'version_banner':first_line}
    return result

@dataclass
class EncodedBPG:
    payload: bytes | None
    side: int | None
    qp: int | None
    attempts: int
    seconds: float

def encode_bpg_caps(source, caps, *, bpgenc='bpgenc', timeout=60):
    """Independent first-fit per cap, sharing identical candidate encodes across caps."""
    from PIL import Image
    enc = resolve_binary(bpgenc)
    if not caps or any(int(cap) != cap or cap <= 0 for cap in caps):
        raise ValueError('Caps must be positive integer native-payload byte limits')
    caps = list(dict.fromkeys(caps));results = {};attempts = 0;started = time.monotonic()
    source = source.convert('RGB')
    with tempfile.TemporaryDirectory(prefix='gap01-bpg-encode-') as td:
        td = Path(td);png = td/'source.png';native = td/'payload.bpg'
        for side in SIDES:
            scale = min(1., side/max(source.size))
            size = tuple(max(1, round(v*scale)) for v in source.size)
            source.resize(size, Image.Resampling.LANCZOS).save(png)
            for qp in QPS:
                command = [enc, '-e', 'x265', '-m', '1', '-b', '8', '-f', '420',
                           '-c', 'ycbcr', '-q', str(qp), '-o', str(native), str(png)]
                proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      timeout=timeout, env={**os.environ, 'OMP_NUM_THREADS':'1'})
                if proc.returncode:
                    raise RuntimeError(f'bpgenc failed ({proc.returncode}): {proc.stderr[-2000:].decode(errors="replace")}')
                payload = native.read_bytes();attempts += 1
                if not payload.startswith(NATIVE_MAGIC):
                    raise RuntimeError('bpgenc output lacks native BPG header')
                for cap in caps:
                    if cap not in results and len(payload) <= cap:
                        results[cap] = EncodedBPG(payload, side, qp, attempts, time.monotonic()-started)
                if len(results) == len(caps):
                    return results
    for cap in caps:
        if cap not in results:
            results[cap] = EncodedBPG(None, None, None, attempts, time.monotonic()-started)
    return results

def decode_bpg(payload, *, bpgdec='bpgdec', timeout=60):
    from PIL import Image
    if not payload.startswith(NATIVE_MAGIC):
        raise ValueError('decode_bpg needs a native BPG payload; remove transport routing byte first')
    dec = resolve_binary(bpgdec)
    with tempfile.TemporaryDirectory(prefix='gap01-bpg-decode-') as td:
        native = Path(td)/'payload.bpg';png = Path(td)/'decoded.png'
        native.write_bytes(payload)
        proc = subprocess.run([dec, '-o', str(png), str(native)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout)
        if proc.returncode:
            raise RuntimeError(f'bpgdec failed ({proc.returncode}): {proc.stderr[-2000:].decode(errors="replace")}')
        with Image.open(png) as im:
            return im.convert('RGB').copy()

def frame(payload):
    if not payload.startswith(NATIVE_MAGIC):
        raise ValueError('Invalid native BPG payload')
    return b'B'+payload

def unframe(packet):
    if not packet.startswith(b'B'+NATIVE_MAGIC):
        raise ValueError('Invalid isolated BPG transport frame')
    return packet[1:]
