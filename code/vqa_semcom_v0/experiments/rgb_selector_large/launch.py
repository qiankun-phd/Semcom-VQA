"""Bounded, sequential EXP-014 launcher with one immutable 24-hour deadline."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_grid import disk_guard, read, save, sha, verify


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def exclusive(path: Path):
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another supervisor owns this experiment') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def commands(args) -> list:
    code, py = Path(__file__).resolve().parent, sys.executable
    common = ['--output',str(args.output),'--protocol',str(args.protocol),'--source-code',str(args.source_code)]
    prep = [py,str(code/'prepare_data.py'),*common,'--project-root',str(args.project_root),'--stage1-root',str(args.stage1_root)]
    features = [py,str(code/'features.py'),*common]
    grid = [py,str(code/'run_grid.py'),*common,'--stage1-root',str(args.stage1_root),
            '--grid-code',str(args.grid_code),'--adapter',str(args.adapter)]
    train = [py,str(code/'train_selectors.py'),*common,'--data-root',str(args.output)]
    evaluate = [py,str(code/'evaluate_test.py'),*common]
    return [
        ['prepare_trainval',[*prep,'--phase','trainval'],'frozen_data.json'],
        ['features_trainval',[*features,'--phase','trainval'],'features_complete.json'],
        ['codec_smoke',[*grid,'--stage','encode','--smoke'],'grid_trainval/encoding_smoke_complete.json'],
        ['vlm_smoke',[*grid,'--stage','infer','--smoke'],'inference_smoke_complete.json'],
        ['codec_trainval',[*grid,'--stage','encode'],'grid_trainval/encoding_complete.json'],
        ['vlm_trainval',[*grid,'--stage','infer'],'supervision_complete.json'],
        ['train_selectors',train,'training_complete.json'],
        ['prepare_test',[*prep,'--phase','test'],'test_data_frozen.json'],
        ['features_test',[*features,'--phase','test'],'test_features_complete.json'],
        ['codec_test',[*grid,'--stage','encode','--phase','test'],'grid_test/encoding_complete.json'],
        ['vlm_test',[*grid,'--stage','infer','--phase','test'],'test_inference_complete.json'],
        ['evaluate_test',evaluate,'test_report.json'],
    ]


def run_step(command: list, logfile: Path, seconds_left: float) -> None:
    if seconds_left <= 0:
        raise TimeoutError('The immutable 24-hour run budget is exhausted')
    with logfile.open('ab') as handle:
        handle.write(f'\nSTART {utc()}\n'.encode())
        handle.flush()
        process = subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,start_new_session=True)
        try:
            code = process.wait(timeout=seconds_left)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
            raise
    if code:
        raise RuntimeError(f'Stage exited {code}; inspect {logfile.name}; no automatic retry')


def validate_completion(name: str, path: Path) -> None:
    value = read(path)
    expected = {'codec_smoke':('representations',18),'vlm_smoke':('records',54),
                'codec_trainval':('representations',18000),'vlm_trainval':('records',54000),
                'codec_test':('representations',7200),'vlm_test':('records',21600)}
    if name in expected:
        key,count = expected[name]
        if value.get(key) != count:
            raise ValueError(f'Incorrect completed quantity for {name}')


def launch(args) -> dict:
    output = args.output
    output.mkdir(parents=True,exist_ok=True)
    protocol = read(args.protocol)
    if protocol['experiment_id'] != 'EXP-014' or protocol['resources']['maximum_total_wallclock_seconds'] != 86400:
        raise ValueError('Unregistered large experiment')
    with exclusive(output/'supervisor.lock'):
        code = Path(__file__).resolve().parent
        paths = sorted(code.glob('*.py'))+[args.protocol]
        stage_commands = commands(args)
        fingerprint = {'sha256':{str(p):sha(p) for p in paths},'commands':stage_commands,
                       'python':sys.executable,'wallclock_limit_seconds':86400}
        frozen_path = output/'supervisor_frozen.json'
        if frozen_path.exists():
            frozen = read(frozen_path)
            if frozen['fingerprint'] != fingerprint:
                raise ValueError('Run code/protocol/commands changed; resume refused')
        else:
            frozen = {'fingerprint':fingerprint,'started_utc':utc(),'deadline_epoch':time.time()+86400}
            save(frozen_path,frozen)
        receipt_path = output/'stage_receipts.json'
        receipts = read(receipt_path) if receipt_path.exists() else {}
        name = 'preflight'
        try:
            for name,command,artifact in stage_commands:
                verify(fingerprint['sha256'])
                disk_guard(output,protocol)
                result_path = output/artifact
                if name in receipts:
                    if receipts[name]['sha256'] != sha(result_path):
                        raise ValueError(f'Previously completed stage artifact changed: {name}')
                    continue
                save(output/'supervisor_status.json',{'state':'RUNNING','stage':name,'pid':os.getpid(),
                    'completed_stages':list(receipts),'updated_utc':utc(),'deadline_epoch':frozen['deadline_epoch'],
                    'old_test300_opened':False,'energy_measured':False})
                run_step(command,output/f'{name}.log',frozen['deadline_epoch']-time.time())
                validate_completion(name,result_path)
                receipts[name] = {'artifact':artifact,'sha256':sha(result_path),'completed_utc':utc()}
                save(receipt_path,receipts)
            result = {'state':'COMPLETE','completed_stages':list(receipts),'updated_utc':utc(),
                      'old_test300_opened':False,'full_cost_gate':'PENDING','energy_measured':False,
                      'test_report_sha256':sha(output/'test_report.json')}
            save(output/'supervisor_status.json',result)
            return result
        except BaseException as error:
            save(output/'supervisor_status.json',{'state':'FAILED','stage':name,'updated_utc':utc(),
                 'completed_stages':list(receipts),'error':str(error),'automatic_retry':False,
                 'deadline_epoch':frozen['deadline_epoch'],'old_test300_opened':False})
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output','protocol','source-code','project-root','stage1-root','grid-code','adapter'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--detach',action='store_true')
    args = parser.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):
            setattr(args,key,value.resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    if args.detach:
        command = [sys.executable,str(Path(__file__).resolve()),*[v for v in sys.argv[1:] if v != '--detach']]
        with (args.output/'launcher.log').open('ab') as handle:
            process = subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        receipt = {'pid':process.pid,'dispatched_utc':utc(),'status_file':str(args.output/'supervisor_status.json')}
        save(args.output/'launch_receipt.json',receipt)
        print(json.dumps(receipt),flush=True)
    else:
        print(json.dumps(launch(args)),flush=True)


if __name__ == '__main__':
    main()
