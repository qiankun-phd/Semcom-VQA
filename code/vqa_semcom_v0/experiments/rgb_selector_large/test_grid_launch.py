"""Dependency-light safety tests; no experiment images or labels are opened."""
import argparse
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import launch
import run_grid as grid


class GridTests(unittest.TestCase):
    def test_unfrozen_test_fails_before_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            opened = []

            def fail(path):
                opened.append(path.name)
                raise FileNotFoundError(path)

            with patch.object(grid,'read',side_effect=fail), self.assertRaises(FileNotFoundError):
                grid.dataset_rows(output,output/'protocol.json','test')
            self.assertEqual(opened,['controller_frozen.json'])

    def test_controller_guard_checks_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output/'weight.pt'
            source.write_bytes(b'fixture')
            upstream = {}
            for name in ('codec_frozen.json', 'inference_frozen.json'):
                path = output/'grid_trainval'/name
                grid.save(path, {'phase':'trainval','sha256':{str(source):grid.sha(source)}})
                upstream[str(path.resolve())] = grid.sha(path)
            grid.save(output/'controller_frozen.json',{'state':'FROZEN_BEFORE_TEST',
                      'sha256':{str(source):grid.sha(source)},'source_inputs_sha256':upstream})
            self.assertEqual(grid.controller_guard(output),grid.sha(output/'controller_frozen.json'))
            source.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                grid.controller_guard(output)

    def test_controller_rejects_unpinned_trainval_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output/'weight.pt'
            source.write_bytes(b'fixture')
            for name in ('codec_frozen.json', 'inference_frozen.json'):
                grid.save(output/'grid_trainval'/name, {'phase':'trainval','sha256':{str(source):grid.sha(source)}})
            grid.save(output/'controller_frozen.json', {'state':'FROZEN_BEFORE_TEST','sha256':{str(source):grid.sha(source)}})
            with self.assertRaisesRegex(ValueError, 'not pinned'):
                grid.controller_guard(output)

    def test_freeze_refuses_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output/'input.json'
            source.write_text('{}')
            fingerprint = {'sha256':{str(source):grid.sha(source)},'phase':'trainval'}
            grid.freeze(output/'frozen.json',fingerprint)
            grid.freeze(output/'frozen.json',fingerprint)
            with self.assertRaises(ValueError):
                grid.freeze(output/'frozen.json',{**fingerprint,'phase':'test'})
            source.write_text('[]')
            with self.assertRaises(ValueError):
                grid.freeze(output/'frozen.json',fingerprint)

    def test_disk_floor(self):
        with patch.object(grid.shutil,'disk_usage',return_value=argparse.Namespace(free=1)):
            with self.assertRaises(RuntimeError):
                grid.disk_guard(Path('.'),{'resources':{'minimum_free_disk_gb':8}})

    def test_phase_and_answer_field_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(ValueError):
                grid.dataset_rows(output,output/'p.json','anything')
            protocol = {'seed':1,'data':{'per_type':{'train':1,'validation':1},'tasks':['one']}}
            grid.save(output/'p.json',protocol)
            grid.save(output/'train_manifest.json',[{'id':'row1','image_id':1,'question_type':'one','answer':'no'}])
            grid.save(output/'validation_manifest.json',[])
            grid.save(output/'image_hashes.json',{})
            files = ['train_manifest.json','validation_manifest.json','image_hashes.json']
            grid.save(output/'frozen_data.json',{'protocol_sha256':grid.sha(output/'p.json'),
                      'sha256':{str((output/name).resolve()):grid.sha(output/name) for name in files}})
            with self.assertRaisesRegex(ValueError,'label-free'):
                grid.dataset_rows(output,output/'p.json','trainval')


class LauncherTests(unittest.TestCase):
    def args(self):
        return argparse.Namespace(**{name:Path('/fixture')/name for name in
            ('output','protocol','source_code','project_root','stage1_root','grid_code','adapter')})

    def test_stage_order_and_independent_test_outputs(self):
        stages = launch.commands(self.args())
        names = [s[0] for s in stages]
        self.assertEqual(len(names),12)
        self.assertLess(names.index('train_selectors'),names.index('prepare_test'))
        self.assertLess(names.index('codec_smoke'),names.index('codec_trainval'))
        self.assertLess(names.index('vlm_smoke'),names.index('vlm_trainval'))
        self.assertNotEqual(stages[5][2],stages[10][2])
        self.assertEqual(names[-1],'evaluate_test')

    def test_quantity_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'complete.json'
            grid.save(path,{'records':54000})
            launch.validate_completion('vlm_trainval',path)
            with self.assertRaises(ValueError):
                launch.validate_completion('vlm_test',path)

    def test_duplicate_supervisor_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'lock'
            with launch.exclusive(path):
                with self.assertRaises(RuntimeError):
                    with launch.exclusive(path):
                        pass

    def test_exhausted_deadline_has_no_process(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(launch.subprocess,'Popen') as popen:
            with self.assertRaises(TimeoutError):
                launch.run_step(['never'],Path(directory)/'log',0)
            popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
