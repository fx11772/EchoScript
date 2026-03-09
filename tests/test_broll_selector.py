import tempfile
import unittest
from pathlib import Path

from tiktok_pipeline.broll_selector import build_clip_plan, extract_keywords, score_broll_file
from tiktok_pipeline.models import ScriptDraft, ScriptLine
from tiktok_pipeline.utils import discover_broll_files


class TestBrollSelector(unittest.TestCase):
    def _sample_script(self) -> ScriptDraft:
        return ScriptDraft(
            niche="fitness motivation",
            lines=[
                ScriptLine(label="HOOK", text="Fitness secrets change everything today"),
                ScriptLine(label="LINE", text="Discipline builds confidence every morning"),
                ScriptLine(label="LINE", text="Motivation follows your daily action"),
                ScriptLine(label="PAYOFF", text="Consistency creates lasting personal transformation"),
                ScriptLine(label="CTA", text="Comment follow and share now"),
            ],
        )

    def test_extract_keywords_returns_core_terms(self):
        keywords = extract_keywords(self._sample_script())
        self.assertIn("fitness", keywords)
        self.assertIn("discipline", keywords)
        self.assertIn("motivation", keywords)

    def test_score_broll_file_matches_filename_terms(self):
        file_path = Path("fitness_morning_motivation.mp4")
        score, matched = score_broll_file(file_path, {"fitness", "motivation", "travel"})

        self.assertEqual(score, 2)
        self.assertEqual(matched, {"fitness", "motivation"})

    def test_build_clip_plan_prefers_keyword_matched_clip(self):
        script = self._sample_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            matched = root / "fitness_motivation_broll.mp4"
            generic = root / "random_city_walk.mp4"
            matched.touch()
            generic.touch()

            plan = build_clip_plan(script, [generic, matched])

        self.assertTrue(plan)
        self.assertEqual(plan[0].clip_path, matched)

    def test_build_clip_plan_uses_fallback_when_no_matches(self):
        script = ScriptDraft(
            niche="rare niche",
            lines=[
                ScriptLine(label="HOOK", text="Obscure phrase no filename matches"),
                ScriptLine(label="LINE", text="Another unusual caption phrase"),
                ScriptLine(label="LINE", text="Third unusual caption phrase"),
                ScriptLine(label="PAYOFF", text="Payoff phrase remains uncommon"),
                ScriptLine(label="CTA", text="Comment follow and share now"),
            ],
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "sunset_city.mp4"
            second = root / "river_drone.mp4"
            first.touch()
            second.touch()

            plan = build_clip_plan(script, [first, second])

        self.assertEqual(plan[0].clip_path, first)
        self.assertEqual(plan[1].clip_path, second)

    def test_build_clip_plan_prefers_niche_subfolder_on_tie(self):
        script = ScriptDraft(
            niche="luxury lifestyle",
            lines=[
                ScriptLine(label="HOOK", text="Luxury cars define your image"),
                ScriptLine(label="LINE", text="Lifestyle details shape status quickly"),
                ScriptLine(label="LINE", text="Success looks obvious from distance"),
                ScriptLine(label="PAYOFF", text="Presence changes every room instantly"),
                ScriptLine(label="CTA", text="Comment follow and share now"),
            ],
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            niche_dir = root / "luxury_lifestyle"
            generic_dir = root / "misc"
            niche_dir.mkdir()
            generic_dir.mkdir()
            preferred = niche_dir / "luxury_cars.mp4"
            generic = generic_dir / "luxury_cars.mp4"
            preferred.touch()
            generic.touch()

            plan = build_clip_plan(script, [generic, preferred])

        self.assertEqual(plan[0].clip_path, preferred)

    def test_discover_broll_files_searches_recursively(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            nested = root / "luxury_lifestyle" / "cars"
            nested.mkdir(parents=True)
            top_level = root / "overview.mp4"
            nested_video = nested / "supercar.mp4"
            ignored = nested / "notes.txt"
            top_level.touch()
            nested_video.touch()
            ignored.write_text("ignore", encoding="utf-8")

            files = discover_broll_files(root)

        self.assertEqual(set(files), {top_level, nested_video})


if __name__ == "__main__":
    unittest.main()
