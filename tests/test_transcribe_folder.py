import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def load_module():
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
    openai.APIStatusError = type("APIStatusError", (Exception,), {})
    openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
    openai.RateLimitError = type("RateLimitError", (Exception,), {})
    dotenv = types.ModuleType("dotenv")
    dotenv.dotenv_values = lambda _path: {}
    with patch.dict(sys.modules, {"openai": openai, "dotenv": dotenv}):
        spec = importlib.util.spec_from_file_location("transcribe_folder", Path(__file__).parents[1] / "transcribe_folder.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


transcribe_folder = load_module()


class ApiAndRenderingTests(unittest.TestCase):
    def test_diarization_request_uses_expected_options(self):
        client = MagicMock()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "meeting.m4a"
            path.write_bytes(b"audio")
            transcribe_folder.transcribe_audio(client, path, "fr")
        kwargs = client.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-4o-transcribe-diarize")
        self.assertEqual(kwargs["response_format"], "diarized_json")
        self.assertEqual(kwargs["chunking_strategy"], "auto")
        self.assertEqual(kwargs["language"], "fr")
        self.assertNotIn("prompt", kwargs)

    def test_auto_language_is_omitted_and_segments_are_rendered(self):
        client = MagicMock()
        client.audio.transcriptions.create.return_value = SimpleNamespace(segments=[
            SimpleNamespace(start=0.2, speaker="speaker_0", text=" Bonjour."),
            SimpleNamespace(start=10.5, speaker="speaker_0", text="La réunion peut commencer."),
            SimpleNamespace(start=65.9, speaker="speaker_1", text="Salut !"),
            SimpleNamespace(start=130, speaker="speaker_0", text="À bientôt."),
        ])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "meeting.m4a"
            path.write_bytes(b"audio")
            result = transcribe_folder.transcribe_audio(client, path, None)
        self.assertNotIn("language", client.audio.transcriptions.create.call_args.kwargs)
        self.assertEqual(
            transcribe_folder.format_diarized_transcript(result),
            "[00:00] Speaker A: Bonjour.\n[00:10] La réunion peut commencer.\n"
            "\n[01:05] Speaker B: Salut !\n\n[02:10] Speaker A: À bientôt.",
        )


class PreparationTests(unittest.TestCase):
    def test_acceptable_file_is_used_directly(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "meeting.m4a"
            source.write_bytes(b"audio")
            path, temporary = transcribe_folder.prepare_audio_for_upload(source, Path(temp) / "prepared")
        self.assertEqual(path, source)
        self.assertFalse(temporary)

    def test_oversized_file_is_reencoded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "meeting.wav"
            source.write_bytes(b"x" * 100)
            output = root / "prepared" / "meeting.diarized.m4a"
            def fake_run(command, **_kwargs):
                if command[0] == "ffprobe":
                    return subprocess.CompletedProcess(command, 0, "120.0\n", "")
                output.parent.mkdir(exist_ok=True)
                output.write_bytes(b"small")
                return subprocess.CompletedProcess(command, 0, "", "")
            with patch.object(transcribe_folder, "MAX_UPLOAD_BYTES", 50), patch.object(
                transcribe_folder.subprocess, "run", side_effect=fake_run
            ) as run:
                path, temporary = transcribe_folder.prepare_audio_for_upload(source, root / "prepared")
        self.assertEqual(path, output)
        self.assertTrue(temporary)
        self.assertEqual(run.call_count, 2)
        self.assertIn("ffmpeg", run.call_args_list[1].args[0])

    def test_reencode_that_remains_too_large_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "meeting.wav"
            source.write_bytes(b"x" * 100)
            output = root / "prepared" / "meeting.diarized.m4a"
            def fake_run(command, **_kwargs):
                if command[0] == "ffprobe":
                    return subprocess.CompletedProcess(command, 0, "120.0\n", "")
                output.parent.mkdir(exist_ok=True)
                output.write_bytes(b"x" * 100)
                return subprocess.CompletedProcess(command, 0, "", "")
            with patch.object(transcribe_folder, "MAX_UPLOAD_BYTES", 50), patch.object(
                transcribe_folder.subprocess, "run", side_effect=fake_run
            ):
                with self.assertRaisesRegex(transcribe_folder.AudioPreparationError, "still too large"):
                    transcribe_folder.prepare_audio_for_upload(source, root / "prepared")


class RetryAndBatchTests(unittest.TestCase):
    def test_retries_transient_error_then_succeeds(self):
        error = transcribe_folder.APIConnectionError("network unavailable")
        with patch.object(transcribe_folder, "transcribe_audio", side_effect=[error, "result"]) as transcribe, patch.object(
            transcribe_folder.time, "sleep"
        ) as sleep:
            self.assertEqual(transcribe_folder.transcribe_audio_with_retry(object(), Path("audio.m4a"), None), "result")
        self.assertEqual(transcribe.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_batch_continues_after_failure_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "failed.m4a").write_bytes(b"audio")
            (input_dir / "successful.m4a").write_bytes(b"audio")
            with patch.object(transcribe_folder, "dotenv_values", return_value={"OPENAI_API_KEY": "test"}), patch.object(
                sys, "argv", ["transcribe_folder.py", str(input_dir), "--prompt", "ignored"]
            ), patch.object(transcribe_folder, "OpenAI", return_value=object()), patch.object(
                transcribe_folder, "transcribe_file", side_effect=[RuntimeError("unavailable"), None]
            ) as transcribe, patch("builtins.print") as output:
                result = transcribe_folder.main()
        self.assertEqual(result, 1)
        self.assertEqual(transcribe.call_count, 2)
        self.assertTrue(any("--prompt is ignored" in str(call) for call in output.call_args_list))

    def test_main_rejects_shell_key_when_dotenv_key_is_missing(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "shell-key"}, clear=True), patch.object(
            transcribe_folder, "dotenv_values", return_value={}
        ), patch.object(sys, "argv", ["transcribe_folder.py", "unused"]), patch.object(transcribe_folder, "OpenAI") as client:
            self.assertEqual(transcribe_folder.main(), 1)
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
