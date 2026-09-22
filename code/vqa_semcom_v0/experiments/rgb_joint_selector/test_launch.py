import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    from .launch import exclusive, run_step, verify_completion
except ImportError:
    from launch import exclusive, run_step, verify_completion


class SupervisorTests(unittest.TestCase):
    def test_success_and_local_log(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "stage.log"
            run_step([sys.executable, "-c", "print('local only')"], log, 10)
            self.assertIn("local only", log.read_text())

    def test_nonzero_stage_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "Stage exited 3"):
                run_step([sys.executable, "-c", "raise SystemExit(3)"], Path(folder) / "log", 10)

    def test_bounded_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(subprocess.TimeoutExpired):
                run_step([sys.executable, "-c", "import time; time.sleep(5)"], Path(folder) / "log", .1)
            with self.assertRaises(TimeoutError):
                run_step([sys.executable, "-c", "pass"], Path(folder) / "log", 0)

    def test_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lock"
            with exclusive(path):
                with self.assertRaisesRegex(RuntimeError, "already owns"):
                    with exclusive(path):
                        self.fail("Duplicate acquired lock")
            with exclusive(path):
                pass

    def test_completion_rejects_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "supervision_complete.json").write_text(json.dumps({"state": "SUPERVISION_COMPLETE", "records": 54}))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                verify_completion("supervision", output)


if __name__ == "__main__":
    unittest.main()
