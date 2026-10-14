"""Exercise gallery's real path/status handling without importing or rendering scenes.

Run with: python3 -m unittest discover -s tests -v
"""
from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


GALLERY_SOURCE = Path(__file__).resolve().parents[1] / "gallery.py"


class GalleryOutputPathsTests(unittest.TestCase):
    def check_output_paths(self, output_kind):
        for outcome in (
            "pass", "skip", "fail_nonzero", "fail_missing_output", "force",
            "render_exception",
        ):
            with self.subTest(output=output_kind, outcome=outcome):
                self.check_run(output_kind, outcome)

    def check_run(self, output_kind, outcome):
        with tempfile.TemporaryDirectory(prefix="gallery paths ") as temporary:
            parent = Path(temporary).resolve()
            root = parent / "kalos repo"
            root.mkdir()
            source = root / "gallery.py"
            shutil.copyfile(GALLERY_SOURCE, source)
            registry = ModuleType("render")
            registry.SCENES = {"sample": {}}
            spec = importlib.util.spec_from_file_location("gallery_under_test", source)
            gallery = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {"render": registry}):
                spec.loader.exec_module(gallery)

            if output_kind == "default":
                argument = None
                out_dir = root / "out"
                log_label = "out/_logs/sample.log"
            elif output_kind == "relative":
                argument = "reels con espacios"
                out_dir = root / argument
                log_label = "reels con espacios/_logs/sample.log"
            elif output_kind == "absolute_internal":
                out_dir = root / "salida interna"
                argument = str(out_dir)
                log_label = "salida interna/_logs/sample.log"
            elif output_kind == "absolute_external":
                out_dir = parent / "salida externa"
                argument = str(out_dir)
                log_label = str(out_dir / "_logs" / "sample.log")
            elif output_kind == "common_prefix_sibling":
                out_dir = parent / "kalos repo-extra"
                argument = str(out_dir)
                log_label = str(out_dir / "_logs" / "sample.log")
            elif output_kind == "relative_external":
                argument = "../salida externa"
                out_dir = root / argument
                log_label = "../salida externa/_logs/sample.log"
            else:
                raise AssertionError(output_kind)

            destination = out_dir / "sample.mp4"
            log_path = out_dir / "_logs" / "sample.log"
            if outcome in ("skip", "force"):
                destination.parent.mkdir(parents=True)
                destination.write_bytes(b"existing video")

            calls = []

            def fake_run(command, *, cwd, stdout, stderr, check):
                calls.append(command)
                self.assertEqual(command, [
                    sys.executable, str(root / "render.py"), "sample", "12.0",
                    "30", str(destination),
                ])
                self.assertEqual(cwd, root)
                self.assertEqual(Path(stdout.name), log_path)
                self.assertEqual(stderr, subprocess.STDOUT)
                self.assertFalse(check)
                stdout.write("simulated renderer output\n")
                if outcome == "render_exception":
                    raise OSError("simulated renderer error")
                if outcome != "fail_missing_output":
                    destination.write_bytes(b"simulated video")
                return subprocess.CompletedProcess(
                    command, 7 if outcome == "fail_nonzero" else 0,
                )

            argv = ["sample", "--jobs", "1"]
            if argument is not None:
                argv.extend(["--out", argument])
            if outcome == "force":
                argv.append("--force")
            other_cwd = parent / "otro cwd"
            other_cwd.mkdir()
            original_cwd = Path.cwd()
            output = io.StringIO()
            try:
                os.chdir(other_cwd)
                with patch.object(gallery.subprocess, "run", side_effect=fake_run):
                    with redirect_stdout(output):
                        code = gallery.main(argv)
            finally:
                os.chdir(original_cwd)

            expected_status = {
                "pass": "PASS", "skip": "SKIP", "fail_nonzero": "FAIL",
                "fail_missing_output": "FAIL", "force": "PASS",
                "render_exception": "FAIL",
            }[outcome]
            self.assertEqual(code, 1 if expected_status == "FAIL" else 0)
            self.assertEqual(len(calls), 0 if outcome == "skip" else 1)
            lines = output.getvalue().splitlines()
            self.assertEqual(len(lines), 1)
            if outcome == "render_exception":
                self.assertEqual(lines[0], "FAIL sample: simulated renderer error")
            else:
                self.assertRegex(lines[0], rf"^{expected_status} sample \(\d+\.\ds\) log=")
                self.assertEqual(lines[0].split(" log=", 1)[1], log_label)
            if outcome == "skip":
                self.assertEqual(destination.read_bytes(), b"existing video")
                self.assertFalse(log_path.exists())
            else:
                self.assertEqual(log_path.read_text(), "simulated renderer output\n")
                if outcome in ("fail_missing_output", "render_exception"):
                    self.assertFalse(destination.exists())
                else:
                    self.assertEqual(destination.read_bytes(), b"simulated video")
            # Relative --out must remain anchored to the script even from another cwd.
            self.assertEqual(list(other_cwd.iterdir()), [])

    def test_default_output(self):
        self.check_output_paths("default")

    def test_relative_output_with_spaces(self):
        self.check_output_paths("relative")

    def test_absolute_internal_output_with_spaces(self):
        self.check_output_paths("absolute_internal")

    def test_absolute_external_output_with_spaces(self):
        self.check_output_paths("absolute_external")

    def test_common_prefix_sibling_is_external(self):
        self.check_output_paths("common_prefix_sibling")

    def test_relative_external_output(self):
        self.check_output_paths("relative_external")


if __name__ == "__main__":
    unittest.main()
