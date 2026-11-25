import io
import json
import os
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from backend.test_voice_pilot import fixture, temporary_directory
from backend.voice_research import audition
from backend.voice_research.pilot import PilotError
from backend.voice_research.review import ReviewStore
from backend.voice_research.cloud_audition import ensure_mapping_link


class AuditionExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.root = Path(self.temp.name)
        self.manifest = fixture(self.root)
        self.folder = self.root / "waxal-ibo-v1"
        self.store = ReviewStore(self.folder, self.manifest)

    def tearDown(self):
        self.temp.cleanup()

    def save_all(self):
        for clip in self.manifest["clips"]:
            self.store.save({"clip_id": clip["id"], "audio_sha256": clip["sha256"],
                             "intelligibility": 5, "pronunciation": 4, "cadence": 3,
                             "transcript_match": "yes", "notes": "Private listening note, not for cloud export"})

    def export(self):
        with patch.object(audition, "ensure_space"), patch("sys.stdout", io.StringIO()):
            return audition.export(self.root)

    def test_export_preserves_published_text_and_excludes_audio_and_notes(self):
        self.save_all()
        original = (self.folder / "reviews.json").read_bytes()
        target = self.export()
        report = json.loads((target / "review-summary.json").read_text())
        self.assertEqual(report["averages_out_of_5"], {"intelligibility": 5, "pronunciation": 4, "cadence": 3})
        cases = json.loads((target / "cases.json").read_text())
        self.assertEqual([c["text"] for c in cases["cases"]], [c["transcript"] for c in self.manifest["clips"]])
        notebook_text = (target / "Anyanwu-voice-audition.ipynb").read_text()
        self.assertNotIn("Private listening note", notebook_text)
        notebook = json.loads(notebook_text)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "exported-cell", "exec")
        run_cell = notebook["cells"][3]
        namespace = {}
        exec("".join(run_cell["source"][:2]), namespace)
        self.assertEqual(namespace["package"], cases)
        self.assertEqual(list(target.glob("*.mp3")) + list(target.glob("*.wav")), [])
        self.assertEqual((self.folder / "reviews.json").read_bytes(), original)
        self.assertEqual(self.export(), target)

    def test_incomplete_feedback_is_not_promoted(self):
        with self.assertRaises(PilotError):
            self.export()
        self.assertFalse((self.root / "auditions").exists())

    def test_uncertain_transcripts_block_same_sentence_audition(self):
        self.save_all()
        self.store.save(self.store.snapshot()["clip-01"] | {"transcript_match": "unsure"})
        with self.assertRaises(PilotError):
            self.export()

    def test_changed_feedback_preserves_existing_snapshot(self):
        self.save_all()
        target = self.export()
        before = (target / "review-summary.json").read_bytes()
        self.store.save(self.store.snapshot()["clip-01"] | {"cadence": 2})
        with self.assertRaises(PilotError):
            self.export()
        self.assertEqual((target / "review-summary.json").read_bytes(), before)

    def test_mac_guards_stop_before_gpu_imports_or_package_install(self):
        with patch("platform.system", return_value="Darwin"):
            with self.assertRaises(SystemExit):
                runpy.run_path(str(audition.HERE / "cloud_audition.py"), run_name="__main__")
            with self.assertRaises(RuntimeError):
                runpy.run_path(str(audition.HERE / "colab_setup.py"), run_name="__main__")

    def test_new_colab_kernels_bootstrap_separate_pinned_python(self):
        real_path = Path
        source = str(audition.HERE / "colab_setup.py")
        for kernel in ((3, 13, 0), (3, 14, 0)):
            with self.subTest(kernel=kernel), temporary_directory() as directory:
                remote = real_path(directory)
                commands = []
                def mapped_path(value):
                    value = str(value)
                    return remote / value.removeprefix("/content").lstrip("/") if value.startswith("/content") else real_path(value)
                class Process:
                    returncode = 0
                    def __init__(self, command, **kwargs):
                        commands.append((command, kwargs["env"]))
                        if "uv==0.9.13" in command:
                            uv = remote / "anyanwu-voice-audition-v1/bootstrap/bin/uv"
                            uv.parent.mkdir(parents=True, exist_ok=True)
                            uv.write_text("fixture only")
                    def poll(self):
                        return 0
                def run(command, **kwargs):
                    if command[0] == "nvidia-smi":
                        return SimpleNamespace(stdout="Tesla T4, 15360", returncode=0)
                    if "-c" in command:
                        return SimpleNamespace(stdout="3.11.14\n", returncode=0)
                    return SimpleNamespace(stdout="", returncode=0)
                with patch("platform.system", return_value="Linux"), patch("pathlib.Path", side_effect=mapped_path), \
                        patch("importlib.util.find_spec", return_value=object()), patch("sys.version_info", kernel), \
                        patch("shutil.disk_usage", return_value=SimpleNamespace(free=40*1024**3)), \
                        patch("subprocess.Popen", Process), patch("subprocess.run", side_effect=run), \
                        patch.dict(os.environ, {"PYTHONPATH": "host-packages", "OPENAI_API_KEY": "fixture-never-export"}), \
                        patch("sys.stdout", io.StringIO()):
                    result = runpy.run_path(source, run_name="__main__")
                self.assertEqual(result["version"], "3.11.14")
                venv = next(command for command, _ in commands if "venv" in command)
                self.assertEqual(venv[venv.index("--python")+1], "3.11.14")
                self.assertTrue(any(command[0].endswith("venv-py311/bin/python") and "chatterbox-tts==0.1.7" in command for command, _ in commands))
                self.assertTrue(all("PYTHONPATH" not in env and "OPENAI_API_KEY" not in env for _, env in commands))

    def test_pinned_mapping_reuses_existing_cache_link(self):
        mapping = self.root / "pinned.json"
        mapping.write_text("mapping fixture")
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        ensure_mapping_link(checkpoint, mapping)
        ensure_mapping_link(checkpoint, mapping)
        self.assertEqual((checkpoint / "Cangjie5_TC.json").resolve(), mapping)

    def test_wrong_cached_mapping_is_preserved_and_refused(self):
        mapping = self.root / "pinned.json"
        mapping.write_text("pinned fixture")
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        target = checkpoint / "Cangjie5_TC.json"
        target.write_text("different mapping")
        with self.assertRaises(RuntimeError):
            ensure_mapping_link(checkpoint, mapping)
        self.assertEqual(target.read_text(), "different mapping")


if __name__ == "__main__":
    unittest.main()
