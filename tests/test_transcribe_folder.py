import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_module():
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
    openai.APIStatusError = type("APIStatusError", (Exception,), {})
    openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
    openai.RateLimitError = type("RateLimitError", (Exception,), {})
    pydub = types.ModuleType("pydub")
    pydub.AudioSegment = object
    dotenv = types.ModuleType("dotenv")
    dotenv.dotenv_values = lambda _path: {}

    with patch.dict(sys.modules, {"openai": openai, "pydub": pydub, "dotenv": dotenv}):
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


class RetryTests(unittest.TestCase):
    def test_retries_transient_error_then_succeeds(self):
        error = transcribe_folder.APIConnectionError("network unavailable")
        with patch.object(
            transcribe_folder, "transcribe_chunk", side_effect=[error, "transcript"]
        ) as transcribe, patch.object(transcribe_folder.time, "sleep") as sleep:
            result = transcribe_folder.transcribe_chunk_with_retry(
                object(), Path("chunk.m4a"), None, None, 1, 1
            )

        self.assertEqual(result, "transcript")
        self.assertEqual(transcribe.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_non_retryable_error_fails_immediately(self):
        with patch.object(
            transcribe_folder, "transcribe_chunk", side_effect=ValueError("invalid request")
        ) as transcribe, patch.object(transcribe_folder.time, "sleep") as sleep:
            with self.assertRaisesRegex(transcribe_folder.ChunkTranscriptionError, "invalid request"):
                transcribe_folder.transcribe_chunk_with_retry(
                    object(), Path("chunk.m4a"), None, None, 1, 1
                )

        self.assertEqual(transcribe.call_count, 1)
        sleep.assert_not_called()

    def test_retries_server_error(self):
        error = transcribe_folder.APIStatusError("server unavailable")
        error.status_code = 503
        with patch.object(
            transcribe_folder, "transcribe_chunk", side_effect=[error, "transcript"]
        ), patch.object(transcribe_folder.time, "sleep") as sleep:
            result = transcribe_folder.transcribe_chunk_with_retry(
                object(), Path("chunk.m4a"), None, None, 1, 1
            )

        self.assertEqual(result, "transcript")
        sleep.assert_called_once_with(1)

    def test_exhausted_retries_preserve_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_path = root / "meeting.m4a"
            audio_path.write_bytes(b"source")
            out_dir = root / "transcripts"
            chunk = out_dir / "_chunks" / "meeting_chunk_000.m4a"
            chunk.parent.mkdir(parents=True)
            chunk.write_bytes(b"audio")
            error = transcribe_folder.APIConnectionError("network unavailable")

            with patch.object(transcribe_folder, "split_audio_to_chunks", return_value=[chunk]), patch.object(
                transcribe_folder, "transcribe_chunk", side_effect=error
            ), patch.object(transcribe_folder.time, "sleep") as sleep:
                with self.assertRaises(transcribe_folder.ChunkTranscriptionError):
                    transcribe_folder.transcribe_file(object(), audio_path, out_dir, "auto", None)

            self.assertTrue(chunk.exists())
            self.assertFalse((out_dir / "meeting.txt").exists())
            self.assertEqual(sleep.call_args_list, [((1,),), ((2,),), ((4,),)])


class BatchFailureTests(unittest.TestCase):
    def test_batch_continues_after_failure_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "failed.m4a").write_bytes(b"audio")
            (input_dir / "successful.m4a").write_bytes(b"audio")
            out_dir = root / "out"

            def transcribe_side_effect(client, audio_path, *_args):
                if audio_path.name == "failed.m4a":
                    raise RuntimeError("unavailable")

            with patch.object(transcribe_folder, "dotenv_values", return_value={"OPENAI_API_KEY": "test"}), patch.object(
                sys, "argv", ["transcribe_folder.py", str(input_dir), "--out", str(out_dir)]
            ), patch.object(transcribe_folder, "OpenAI", return_value=object()), patch.object(
                transcribe_folder, "transcribe_file", side_effect=transcribe_side_effect
            ) as transcribe:
                result = transcribe_folder.main()

            self.assertEqual(result, 1)
            self.assertEqual(transcribe.call_count, 2)

    def test_main_reads_key_from_dotenv_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "meeting.m4a").write_bytes(b"audio")

            with patch.object(transcribe_folder, "dotenv_values", return_value={"OPENAI_API_KEY": "from-dotenv"}), patch.object(
                sys, "argv", ["transcribe_folder.py", str(input_dir)]
            ), patch.object(transcribe_folder, "OpenAI", return_value=object()) as openai_client, patch.object(
                transcribe_folder, "transcribe_file"
            ):
                result = transcribe_folder.main()

            self.assertEqual(result, 0)
            openai_client.assert_called_once_with(api_key="from-dotenv", max_retries=0)

    def test_main_rejects_shell_key_when_dotenv_key_is_missing(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "shell-key"}, clear=True), patch.object(
            transcribe_folder, "dotenv_values", return_value={}
        ), patch.object(sys, "argv", ["transcribe_folder.py", "unused"]), patch.object(
            transcribe_folder, "OpenAI"
        ) as openai_client:
            result = transcribe_folder.main()

        self.assertEqual(result, 1)
        openai_client.assert_not_called()

    def test_load_api_key_rejects_blank_dotenv_value(self):
        with patch.object(transcribe_folder, "dotenv_values", return_value={"OPENAI_API_KEY": "  "}):
            api_key = transcribe_folder.load_api_key()

        self.assertIsNone(api_key)


if __name__ == "__main__":
    unittest.main()
