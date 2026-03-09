import unittest

from tiktok_pipeline.pexels import (
    build_download_candidates,
    build_download_filename,
    select_best_video_file,
)


class TestPexelsHelpers(unittest.TestCase):
    def test_select_best_video_file_prefers_portrait_mp4(self):
        chosen = select_best_video_file(
            [
                {
                    "link": "https://example.com/landscape.mp4",
                    "file_type": "video/mp4",
                    "width": 1920,
                    "height": 1080,
                },
                {
                    "link": "https://example.com/portrait.mp4",
                    "file_type": "video/mp4",
                    "width": 720,
                    "height": 1280,
                },
                {
                    "link": "https://example.com/portrait.webm",
                    "file_type": "video/webm",
                    "width": 720,
                    "height": 1280,
                },
            ]
        )

        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["link"], "https://example.com/portrait.mp4")

    def test_build_download_candidates_filters_duration_and_maps_fields(self):
        payload = {
            "videos": [
                {
                    "id": 123,
                    "duration": 8,
                    "url": "https://www.pexels.com/video/123/",
                    "video_files": [
                        {
                            "link": "https://player.pexels.com/123.mp4",
                            "file_type": "video/mp4",
                            "width": 720,
                            "height": 1280,
                        }
                    ],
                },
                {
                    "id": 999,
                    "duration": 30,
                    "url": "https://www.pexels.com/video/999/",
                    "video_files": [
                        {
                            "link": "https://player.pexels.com/999.mp4",
                            "file_type": "video/mp4",
                            "width": 720,
                            "height": 1280,
                        }
                    ],
                },
            ]
        }

        candidates = build_download_candidates(payload, "fitness motivation", min_duration=3, max_duration=20)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].video_id, 123)
        self.assertEqual(candidates[0].source_query, "fitness motivation")

    def test_build_download_filename_keeps_query_terms_for_selector(self):
        filename = build_download_filename("fitness motivation", "morning routine", 123, "video/mp4")

        self.assertEqual(filename, "fitness-motivation-morning-routine-pexels-123.mp4")


if __name__ == "__main__":
    unittest.main()
