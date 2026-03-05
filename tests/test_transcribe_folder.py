import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import transcribe_folder as tf


class _FakeChunk:
    def __init__(self, start_ms: int, end_ms: int):
        self.start_ms = start_ms
        self.end_ms = end_ms

    def export(self, path: str, format: str):
        Path(path).touch()


class _FakeAudio:
    def __init__(self, duration_ms: int):
        self.duration_ms = duration_ms

    def __len__(self):
        return self.duration_ms

    def __getitem__(self, slice_obj):
        return _FakeChunk(slice_obj.start, slice_obj.stop)


class TestTranscribeFolder(unittest.TestCase):
    def test_split_audio_to_chunks_creates_expected_chunk_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio_path = root / "talk.m4a"
            audio_path.touch()
            workdir = root / "chunks"
            workdir.mkdir()

            fake_audio = _FakeAudio(duration_ms=25 * 60 * 1000)
            with patch.object(tf.AudioSegment, "from_file", return_value=fake_audio):
                chunks = tf.split_audio_to_chunks(audio_path, workdir)

            self.assertEqual(len(chunks), 3)
            self.assertEqual(
                [p.name for p in chunks],
                ["talk_chunk_000.m4a", "talk_chunk_001.m4a", "talk_chunk_002.m4a"],
            )
            self.assertTrue(all(p.exists() for p in chunks))

    def test_transcribe_chunk_passes_optional_language_and_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            chunk = Path(td) / "c.m4a"
            chunk.write_bytes(b"audio")

            create_mock = MagicMock(return_value=type("Resp", (), {"text": "hello"})())
            client = MagicMock()
            client.audio.transcriptions.create = create_mock

            text = tf.transcribe_chunk(client, chunk, "en", "domain terms")

            self.assertEqual(text, "hello")
            kwargs = create_mock.call_args.kwargs
            self.assertEqual(kwargs["model"], "gpt-4o-mini-transcribe")
            self.assertEqual(kwargs["language"], "en")
            self.assertEqual(kwargs["prompt"], "domain terms")
            self.assertIn("file", kwargs)

    def test_transcribe_chunk_omits_optional_fields_when_none(self):
        with tempfile.TemporaryDirectory() as td:
            chunk = Path(td) / "c.m4a"
            chunk.write_bytes(b"audio")

            create_mock = MagicMock(return_value=type("Resp", (), {"text": "hello"})())
            client = MagicMock()
            client.audio.transcriptions.create = create_mock

            tf.transcribe_chunk(client, chunk, None, None)

            kwargs = create_mock.call_args.kwargs
            self.assertNotIn("language", kwargs)
            self.assertNotIn("prompt", kwargs)

    def test_transcribe_file_writes_merged_transcript(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio = root / "session.mp3"
            audio.touch()
            out_dir = root / "out"
            chunk_paths = [root / "ch1.m4a", root / "ch2.m4a"]
            for c in chunk_paths:
                c.touch()

            with patch.object(tf, "split_audio_to_chunks", return_value=chunk_paths), patch.object(
                tf, "transcribe_chunk", side_effect=["part one", "part two"]
            ) as mock_transcribe_chunk:
                tf.transcribe_file(client=MagicMock(), audio_path=audio, out_dir=out_dir, lang_mode="auto", prompt=None)

            out_file = out_dir / "session.txt"
            self.assertTrue(out_file.exists())
            self.assertEqual(out_file.read_text(encoding="utf-8"), "part one\n\npart two")
            self.assertEqual(mock_transcribe_chunk.call_count, 2)

    def test_main_filters_supported_extensions_and_processes_sorted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            in_dir = root / "in"
            in_dir.mkdir()
            (in_dir / "b.wav").touch()
            (in_dir / "a.mp3").touch()
            (in_dir / "ignore.txt").touch()

            processed = []
            argv = ["transcribe_folder.py", str(in_dir), "--out", str(root / "out"), "--lang", "auto"]

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False), patch.object(
                tf, "OpenAI", side_effect=lambda api_key: {"api_key": api_key}
            ), patch.object(
                tf,
                "transcribe_file",
                side_effect=lambda client, audio_path, out_dir, lang_mode, prompt: processed.append(audio_path.name),
            ), patch.object(
                sys, "argv", argv
            ):
                tf.main()

            self.assertEqual(processed, ["a.mp3", "b.wav"])

    def test_main_exits_when_api_key_missing(self):
        with tempfile.TemporaryDirectory() as td:
            in_dir = Path(td) / "in"
            in_dir.mkdir()

            with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", ["transcribe_folder.py", str(in_dir)]):
                with self.assertRaises(SystemExit) as exc:
                    tf.main()

            self.assertEqual(exc.exception.code, 1)

    def test_main_exits_when_no_audio_files(self):
        with tempfile.TemporaryDirectory() as td:
            in_dir = Path(td) / "in"
            in_dir.mkdir()
            (in_dir / "note.txt").touch()

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False), patch.object(
                tf, "OpenAI", side_effect=lambda api_key: {"api_key": api_key}
            ), patch.object(sys, "argv", ["transcribe_folder.py", str(in_dir)]):
                with self.assertRaises(SystemExit) as exc:
                    tf.main()

            self.assertEqual(exc.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
