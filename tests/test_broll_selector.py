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

    def test_build_clip_plan_is_deterministic_with_seed(self):
        script = self._sample_script()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first_dir = root / "set_a"
            second_dir = root / "set_b"
            first_dir.mkdir()
            second_dir.mkdir()
            first = first_dir / "clip_one.mp4"
            second = second_dir / "clip_two.mp4"
            first.touch()
            second.touch()

            first_plan = build_clip_plan(script, [first, second], seed=11)
            second_plan = build_clip_plan(script, [first, second], seed=11)

        self.assertEqual(
            [item.clip_path for item in first_plan],
            [item.clip_path for item in second_plan],
        )

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

            plan = build_clip_plan(script, [first, second], seed=5)

        self.assertIn(plan[0].clip_path, {first, second})
        self.assertIn(plan[1].clip_path, {first, second})

    def test_build_clip_plan_avoids_immediate_folder_repeat(self):
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
            cars_dir = root / "cars"
            watches_dir = root / "watches"
            cars_dir.mkdir()
            watches_dir.mkdir()
            car_clip = cars_dir / "luxury_cars.mp4"
            watch_clip = watches_dir / "luxury_watch.mp4"
            car_clip.touch()
            watch_clip.touch()

            plan = build_clip_plan(script, [car_clip, watch_clip], seed=3)

        folder_names = [item.clip_path.parent.name for item in plan if item.clip_path is not None]
        for idx in range(1, len(folder_names)):
            self.assertNotEqual(folder_names[idx], folder_names[idx - 1])

    def test_build_clip_plan_spreads_selection_across_relevant_folders(self):
        script = ScriptDraft(
            niche="luxury lifestyle",
            lines=[
                ScriptLine(label="HOOK", text="Luxury lifestyle shifts your image"),
                ScriptLine(label="LINE", text="Luxury details define your status"),
                ScriptLine(label="LINE", text="Lifestyle signals build silent power"),
                ScriptLine(label="LINE", text="Luxury taste separates winners fast"),
                ScriptLine(label="LINE", text="Status symbols control first impressions"),
                ScriptLine(label="PAYOFF", text="Presence changes every room instantly"),
                ScriptLine(label="CTA", text="Comment follow and share now"),
            ],
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cars_dir = root / "luxury_lifestyle" / "cars"
            watches_dir = root / "luxury_lifestyle" / "watches"
            travel_dir = root / "luxury_lifestyle" / "travel"
            cars_dir.mkdir(parents=True)
            watches_dir.mkdir(parents=True)
            travel_dir.mkdir(parents=True)
            car_clip = cars_dir / "luxury_status.mp4"
            watch_clip = watches_dir / "luxury_status.mp4"
            travel_clip = travel_dir / "luxury_status.mp4"
            car_clip.touch()
            watch_clip.touch()
            travel_clip.touch()

            plan = build_clip_plan(script, [car_clip, watch_clip, travel_clip], seed=7)

        folder_names = [item.clip_path.parent.name for item in plan if item.clip_path is not None]
        selected_folders = set(folder_names)
        self.assertEqual(selected_folders, {"cars", "watches", "travel"})
        self.assertLessEqual(folder_names.count("cars"), 3)

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
