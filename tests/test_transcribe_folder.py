import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_module():
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    pydub = types.ModuleType("pydub")
    pydub.AudioSegment = object

    with patch.dict(sys.modules, {"openai": openai, "pydub": pydub}):
        spec = importlib.util.spec_from_file_location(
            "transcribe_folder", Path(__file__).parents[1] / "transcribe_folder.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


transcribe_folder = load_module()


class TranscribeFileCleanupTests(unittest.TestCase):
    def create_chunks(self, directory: Path, stem: str, count: int = 2) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        chunks = []
        for index in range(count):
            path = directory / f"{stem}_chunk_{index:03d}.m4a"
            path.write_bytes(b"audio")
            chunks.append(path)
        return chunks

    def test_success_removes_its_chunks_and_empty_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_path = root / "meeting.m4a"
            audio_path.write_bytes(b"source")
            out_dir = root / "transcripts"
            chunks = self.create_chunks(out_dir / "_chunks", "meeting")

            with patch.object(transcribe_folder, "split_audio_to_chunks", return_value=chunks), patch.object(
                transcribe_folder, "transcribe_chunk", side_effect=["first", "second"]
            ):
                transcribe_folder.transcribe_file(object(), audio_path, out_dir, "auto", None)

            self.assertEqual((out_dir / "meeting.txt").read_text(encoding="utf-8"), "first\n\nsecond")
            self.assertFalse(any(chunk.exists() for chunk in chunks))
            self.assertFalse((out_dir / "_chunks").exists())

    def test_failure_preserves_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_path = root / "meeting.m4a"
            audio_path.write_bytes(b"source")
            out_dir = root / "transcripts"
            chunks = self.create_chunks(out_dir / "_chunks", "meeting")

            with patch.object(transcribe_folder, "split_audio_to_chunks", return_value=chunks), patch.object(
                transcribe_folder, "transcribe_chunk", side_effect=RuntimeError("API unavailable")
            ):
                with self.assertRaisesRegex(RuntimeError, "API unavailable"):
                    transcribe_folder.transcribe_file(object(), audio_path, out_dir, "auto", None)

            self.assertTrue(all(chunk.exists() for chunk in chunks))
            self.assertTrue((out_dir / "_chunks").is_dir())
            self.assertFalse((out_dir / "meeting.txt").exists())

    def test_success_keeps_chunks_from_another_failed_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_path = root / "successful.m4a"
            audio_path.write_bytes(b"source")
            out_dir = root / "transcripts"
            tmp_dir = out_dir / "_chunks"
            successful_chunks = self.create_chunks(tmp_dir, "successful")
            failed_chunks = self.create_chunks(tmp_dir, "failed")

            with patch.object(transcribe_folder, "split_audio_to_chunks", return_value=successful_chunks), patch.object(
                transcribe_folder, "transcribe_chunk", return_value="transcript"
            ):
                transcribe_folder.transcribe_file(object(), audio_path, out_dir, "auto", None)

            self.assertFalse(any(chunk.exists() for chunk in successful_chunks))
            self.assertTrue(all(chunk.exists() for chunk in failed_chunks))
            self.assertTrue(tmp_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
