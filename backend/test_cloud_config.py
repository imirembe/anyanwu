import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.cloud import configure


class CloudConfigTests(unittest.TestCase):
    def test_laptop_cannot_enable_model_downloads(self):
        with patch.object(configure.platform, "system", return_value="Darwin"), patch.object(configure.subprocess, "run") as run:
            with self.assertRaisesRegex(SystemExit, "Nothing was downloaded or changed"):
                configure.main()
            run.assert_not_called()

    def test_cloud_tokens_are_private_unprinted_and_never_overwritten(self):
        Path(".build").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=".build", prefix="cloud-config-check-") as directory:
            source = Path(directory) / "configure.py"
            output = io.StringIO()
            with patch.object(configure, "__file__", str(source)), patch.object(configure.platform, "system", return_value="Linux"), \
                    patch.object(configure.platform, "machine", return_value="x86_64"), patch.object(configure.subprocess, "run"), \
                    contextlib.redirect_stdout(output):
                configure.main()
                env = source.with_name(".env")
                original = env.read_text()
                values = dict(line.split("=", 1) for line in original.splitlines() if "=" in line)
                self.assertEqual(values["ANYANWU_ALLOW_MODEL_DOWNLOAD"], "1")
                for key in ("ANYANWU_ACCESS_TOKEN", "ANYANWU_WORKER_TOKEN"):
                    self.assertGreaterEqual(len(values[key]), 32)
                    self.assertNotIn(values[key], output.getvalue())
                self.assertEqual(os.stat(env).st_mode & 0o777, 0o600)
                with self.assertRaisesRegex(SystemExit, "preserved"):
                    configure.main()
                self.assertEqual(env.read_text(), original)
