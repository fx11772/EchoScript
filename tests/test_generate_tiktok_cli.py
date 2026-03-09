import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import generate_tiktok as gt
from tiktok_pipeline.models import ScriptDraft, ScriptLine


class TestGenerateTikTokCLI(unittest.TestCase):
    def _script(self, niche: str) -> ScriptDraft:
        return ScriptDraft(
            niche=niche,
            lines=[
                ScriptLine(label="HOOK", text="What if luxury rewires status today"),
                ScriptLine(label="LINE", text="Wealth signals speak before words"),
                ScriptLine(label="LINE", text="Luxury habits change your energy"),
                ScriptLine(label="PAYOFF", text="Presence shapes every room instantly"),
                ScriptLine(label="CTA", text="Comment follow and share now"),
            ],
        )

    def test_main_creates_phase_1b_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            broll_dir = root / "broll"
            out_dir = root / "output"
            broll_dir.mkdir()
            (broll_dir / "clip_one.mp4").touch()

            with patch("tiktok_pipeline.pipeline.generate_script", return_value=self._script("fitness motivation")), patch(
                "tiktok_pipeline.pipeline.create_narration_and_alignment",
                return_value=(out_dir / "fake.mp3", out_dir / "fake_alignment.json"),
            ):
                rc = gt.main([
                    "--niche",
                    "fitness motivation",
                    "--broll-dir",
                    str(broll_dir),
                    "--out-dir",
                    str(out_dir),
                ])

            self.assertEqual(rc, 0)
            run_dir = out_dir / "fitness-motivation"
            self.assertTrue((run_dir / "scripts").is_dir())
            self.assertTrue((run_dir / "audio").is_dir())
            self.assertTrue((run_dir / "alignments").is_dir())
            self.assertTrue((run_dir / "plans").is_dir())
            self.assertTrue((run_dir / "renders").is_dir())
            self.assertTrue((run_dir / "temp").is_dir())

            manifest_path = run_dir / "phase_1c_manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["phase"], "1C")
            self.assertEqual(manifest["broll_count"], 1)
            self.assertGreaterEqual(manifest["clip_plan_count"], 1)

    def test_main_returns_error_when_no_broll_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            broll_dir = root / "broll"
            out_dir = root / "output"
            broll_dir.mkdir()
            (broll_dir / "note.txt").write_text("not video", encoding="utf-8")

            rc = gt.main([
                "--niche",
                "finance",
                "--broll-dir",
                str(broll_dir),
                "--out-dir",
                str(out_dir),
            ])
            self.assertEqual(rc, 1)

    def test_main_discovers_nested_broll_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            broll_dir = root / "broll"
            out_dir = root / "output"
            nested = broll_dir / "luxury_lifestyle"
            nested.mkdir(parents=True)
            (nested / "luxury_cars.mp4").touch()

            with patch("tiktok_pipeline.pipeline.generate_script", return_value=self._script("luxury lifestyle")), patch(
                "tiktok_pipeline.pipeline.create_narration_and_alignment",
                return_value=(out_dir / "fake.mp3", out_dir / "fake_alignment.json"),
            ):
                rc = gt.main([
                    "--niche",
                    "luxury lifestyle",
                    "--broll-dir",
                    str(broll_dir),
                    "--out-dir",
                    str(out_dir),
                ])

            self.assertEqual(rc, 0)
            manifest_path = out_dir / "luxury-lifestyle" / "phase_1c_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["broll_count"], 1)

    def test_main_returns_error_for_invalid_duration(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            broll_dir = root / "broll"
            out_dir = root / "output"
            broll_dir.mkdir()
            (broll_dir / "clip.mp4").touch()

            rc = gt.main([
                "--niche",
                "finance",
                "--broll-dir",
                str(broll_dir),
                "--out-dir",
                str(out_dir),
                "--duration-seconds",
                "0",
            ])
            self.assertEqual(rc, 1)

    def test_main_overwrite_removes_existing_run_dir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            broll_dir = root / "broll"
            out_dir = root / "output"
            broll_dir.mkdir()
            (broll_dir / "clip.mp4").touch()

            run_dir = out_dir / "self-growth"
            run_dir.mkdir(parents=True)
            stale_file = run_dir / "stale.txt"
            stale_file.write_text("old", encoding="utf-8")

            with patch("tiktok_pipeline.pipeline.generate_script", return_value=self._script("self growth")), patch(
                "tiktok_pipeline.pipeline.create_narration_and_alignment",
                return_value=(out_dir / "fake.mp3", out_dir / "fake_alignment.json"),
            ):
                rc = gt.main([
                    "--niche",
                    "self growth",
                    "--broll-dir",
                    str(broll_dir),
                    "--out-dir",
                    str(out_dir),
                    "--overwrite",
                ])

            self.assertEqual(rc, 0)
            self.assertFalse(stale_file.exists())
            self.assertTrue((run_dir / "phase_1c_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
