import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tiktok_pipeline.models import ScriptDraft, ScriptLine
from tiktok_pipeline.models import TimedToken
from tiktok_pipeline.tts_align import (
    create_narration_and_alignment,
    normalize_timed_lines,
    script_text,
)


class TestTTSAlign(unittest.TestCase):
    def _script(self) -> ScriptDraft:
        return ScriptDraft(
            niche="luxury lifestyle",
            lines=[
                ScriptLine(label="HOOK", text="What if luxury rewires status today"),
                ScriptLine(label="LINE", text="Wealth signals speak before words"),
                ScriptLine(label="LINE", text="Luxury habits change your energy"),
                ScriptLine(label="PAYOFF", text="Presence shapes every room instantly"),
                ScriptLine(label="CTA", text="Comment follow and share now"),
            ],
        )

    def test_script_text_flattens_lines(self):
        self.assertIn("What if luxury rewires status today", script_text(self._script()))

    def test_normalize_timed_lines_maps_tokens_across_script_lines(self):
        timed_tokens = [
            TimedToken(token="What", start_s=0.0, end_s=0.2),
            TimedToken(token="if", start_s=0.2, end_s=0.4),
            TimedToken(token="luxury", start_s=0.4, end_s=0.7),
            TimedToken(token="rewires", start_s=0.7, end_s=1.0),
            TimedToken(token="status", start_s=1.0, end_s=1.3),
            TimedToken(token="today", start_s=1.3, end_s=1.6),
            TimedToken(token="Wealth", start_s=1.6, end_s=1.9),
            TimedToken(token="signals", start_s=1.9, end_s=2.2),
            TimedToken(token="speak", start_s=2.2, end_s=2.5),
            TimedToken(token="before", start_s=2.5, end_s=2.8),
            TimedToken(token="words", start_s=2.8, end_s=3.1),
        ]

        timed_lines = normalize_timed_lines(self._script(), timed_tokens, duration_s=3.1)

        self.assertEqual(len(timed_lines), 5)
        self.assertEqual(timed_lines[0].label, "HOOK")
        self.assertGreater(len(timed_lines[0].tokens), 0)
        self.assertLessEqual(timed_lines[0].start_s, timed_lines[0].end_s)

    def test_create_narration_and_alignment_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio_dir = root / "audio"
            alignments_dir = root / "alignments"
            audio_dir.mkdir()
            alignments_dir.mkdir()

            speech_response = MagicMock()
            def _stream_to_file(path: str) -> None:
                Path(path).write_bytes(b"fake audio")
            speech_response.stream_to_file.side_effect = _stream_to_file

            transcription_response = {
                "text": script_text(self._script()),
                "duration": 5.0,
                "words": [
                    {"word": "What", "start": 0.0, "end": 0.2},
                    {"word": "if", "start": 0.2, "end": 0.4},
                    {"word": "luxury", "start": 0.4, "end": 0.8},
                    {"word": "rewires", "start": 0.8, "end": 1.2},
                    {"word": "status", "start": 1.2, "end": 1.5},
                    {"word": "today", "start": 1.5, "end": 1.8},
                ],
            }

            client = MagicMock()
            client.audio.speech.create.return_value = speech_response
            client.audio.transcriptions.create.return_value = transcription_response

            narration_path, alignment_path = create_narration_and_alignment(
                script=self._script(),
                audio_dir=audio_dir,
                alignments_dir=alignments_dir,
                voice="alloy",
                client=client,
            )

            self.assertTrue(narration_path.exists())
            self.assertTrue(alignment_path.exists())
            payload = json.loads(alignment_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["voice"], "alloy")
            self.assertEqual(payload["tts_model"], "gpt-4o-mini-tts")
            self.assertTrue(payload["timed_lines"])

    def test_create_narration_and_alignment_requires_api_key_without_client(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio_dir = root / "audio"
            alignments_dir = root / "alignments"
            audio_dir.mkdir()
            alignments_dir.mkdir()

            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(ValueError):
                    create_narration_and_alignment(
                        script=self._script(),
                        audio_dir=audio_dir,
                        alignments_dir=alignments_dir,
                    )


if __name__ == "__main__":
    unittest.main()
