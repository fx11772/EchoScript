import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError


SUPPORTED_EXTS = {".m4a", ".mp3", ".wav", ".aac", ".webm", ".mp4"}
RETRY_DELAYS_SECONDS = (1, 2, 4)
# OpenAI's audio transcription endpoint accepts files up to 25 MiB.  Leave a
# little room below that limit when creating an AAC container.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
TARGET_UPLOAD_BYTES = 24 * 1024 * 1024
# The diarization model currently accepts at most 1,400 seconds per request.
MAX_API_DIARIZATION_SECONDS = 1_400
# Keep a ten-second margin when asking ffmpeg to segment. AAC frame timing can
# still make a produced part a few milliseconds longer than this target.
DIARIZATION_SEGMENT_SECONDS = 1_390
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"


class TranscriptionError(RuntimeError):
    """Raised when an audio file cannot be transcribed after the allowed attempts."""


class AudioPreparationError(RuntimeError):
    """Raised when an oversized recording cannot be made uploadable."""


def load_api_key() -> str | None:
    """Return the OpenAI API key configured in the project .env file only."""
    api_key = dotenv_values(ENV_FILE).get("OPENAI_API_KEY")
    return api_key.strip() if isinstance(api_key, str) and api_key.strip() else None


def audio_duration_seconds(audio_path: Path) -> float:
    """Read duration through ffprobe, which is installed with ffmpeg."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        duration = float(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise AudioPreparationError(
            f"Could not determine the duration of {audio_path.name}; ffmpeg/ffprobe is required: {exc}"
        ) from exc
    if duration <= 0:
        raise AudioPreparationError(f"Could not determine a valid duration for {audio_path.name}")
    return duration


def prepare_audio_for_upload(audio_path: Path, workdir: Path) -> tuple[Path, bool]:
    """Return an uploadable file, re-encoding oversized audio when necessary."""
    if audio_path.stat().st_size <= MAX_UPLOAD_BYTES:
        return audio_path, False

    workdir.mkdir(parents=True, exist_ok=True)
    duration = audio_duration_seconds(audio_path)
    # Reserve a small amount for the M4A container and choose a bitrate that
    # should fit the target. Mono 16 kHz speech remains clear at low bitrates.
    bitrate_kbps = max(8, min(128, int((TARGET_UPLOAD_BYTES * 8) / duration / 1000)))
    prepared_path = workdir / f"{audio_path.stem}.diarized.m4a"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio_path), "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "aac", "-b:a", f"{bitrate_kbps}k", str(prepared_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AudioPreparationError(
            f"Could not re-encode {audio_path.name}; ffmpeg is required: {exc}"
        ) from exc

    if not prepared_path.exists() or prepared_path.stat().st_size > MAX_UPLOAD_BYTES:
        size = prepared_path.stat().st_size if prepared_path.exists() else 0
        raise AudioPreparationError(
            f"{audio_path.name} is still too large after re-encoding ({size} bytes; "
            f"limit: {MAX_UPLOAD_BYTES} bytes)."
        )
    return prepared_path, True


def split_audio_for_diarization(audio_path: Path, workdir: Path) -> tuple[list[tuple[Path, float]], bool]:
    """Split recordings exceeding the model duration limit into uploadable parts."""
    duration = audio_duration_seconds(audio_path)
    if duration <= MAX_API_DIARIZATION_SECONDS:
        return [(audio_path, 0.0)], False

    workdir.mkdir(parents=True, exist_ok=True)
    part_pattern = workdir / f"{audio_path.stem}.part_%03d.m4a"
    for stale_part in workdir.glob(f"{audio_path.stem}.part_*.m4a"):
        try:
            stale_part.unlink()
        except OSError as exc:
            raise AudioPreparationError(f"Could not replace temporary part {stale_part.name}: {exc}") from exc
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio_path), "-vn", "-map", "0:a", "-f", "segment",
                "-segment_time", str(DIARIZATION_SEGMENT_SECONDS), "-reset_timestamps", "1",
                "-c:a", "aac", "-b:a", "64k", str(part_pattern),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AudioPreparationError(f"Could not split {audio_path.name} for diarization: {exc}") from exc

    paths = sorted(workdir.glob(f"{audio_path.stem}.part_*.m4a"))
    if len(paths) < 2:
        raise AudioPreparationError(f"Could not split {audio_path.name} into diarization parts.")

    parts: list[tuple[Path, float]] = []
    offset = 0.0
    for path in paths:
        part_duration = audio_duration_seconds(path)
        if part_duration > MAX_API_DIARIZATION_SECONDS:
            raise AudioPreparationError(f"Diarization part {path.name} exceeds the duration limit.")
        parts.append((path, offset))
        offset += part_duration
    return parts, True


def transcribe_audio(client: OpenAI, audio_path: Path, lang: str | None) -> Any:
    with audio_path.open("rb") as audio_file:
        kwargs: dict[str, Any] = {
            "model": "gpt-4o-transcribe-diarize",
            "file": audio_file,
            "response_format": "diarized_json",
            "chunking_strategy": "auto",
        }
        if lang:
            kwargs["language"] = lang
        return client.audio.transcriptions.create(**kwargs)


def is_retryable_error(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    return isinstance(error, APIStatusError) and 500 <= error.status_code < 600


def transcribe_audio_with_retry(client: OpenAI, audio_path: Path, lang: str | None) -> Any:
    for attempt, delay in enumerate((*RETRY_DELAYS_SECONDS, None), start=1):
        try:
            return transcribe_audio(client, audio_path, lang)
        except Exception as exc:
            if not is_retryable_error(exc) or delay is None:
                raise TranscriptionError(f"{audio_path.name} failed: {exc}") from exc
            print(
                f"  Transient error: {exc}. Retrying in {delay} second(s) "
                f"({attempt}/{len(RETRY_DELAYS_SECONDS)})..."
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def segment_value(segment: Any, name: str) -> Any:
    return segment.get(name) if isinstance(segment, dict) else getattr(segment, name, None)


def speaker_label(index: int) -> str:
    """Convert 0, 1, ... to A, B, ... AA for friendly anonymous labels."""
    letters = ""
    while True:
        letters = chr(ord("A") + index % 26) + letters
        index = index // 26 - 1
        if index < 0:
            return f"Speaker {letters}"


def format_diarized_transcript(result: Any) -> str:
    segments = segment_value(result, "segments")
    if not segments:
        raise TranscriptionError("The API returned no diarized segments.")

    speakers: dict[str, str] = {}
    lines: list[str] = []
    previous_speaker: str | None = None
    for segment in segments:
        raw_speaker = segment_value(segment, "speaker")
        text = segment_value(segment, "text")
        start = segment_value(segment, "start")
        if raw_speaker is None or text is None or start is None:
            raise TranscriptionError("The API returned an invalid diarized segment.")
        key = str(raw_speaker)
        label = speakers.setdefault(key, speaker_label(len(speakers)))
        total_seconds = max(0, int(float(start)))
        timestamp = f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"
        if previous_speaker is not None and key != previous_speaker:
            lines.append("")
        prefix = f"{label}: " if key != previous_speaker else ""
        lines.append(f"[{timestamp}] {prefix}{str(text).strip()}")
        previous_speaker = key
    return "\n".join(lines)


def format_readable_transcript(result: Any, title: str) -> str:
    """Render consecutive segments from one speaker as a readable Markdown turn."""
    segments = segment_value(result, "segments")
    if not segments:
        raise TranscriptionError("The API returned no diarized segments.")

    speakers: dict[str, str] = {}
    turns: list[tuple[str, list[str]]] = []
    for segment in segments:
        raw_speaker = segment_value(segment, "speaker")
        text = segment_value(segment, "text")
        if raw_speaker is None or text is None:
            raise TranscriptionError("The API returned an invalid diarized segment.")
        key = str(raw_speaker)
        label = speakers.setdefault(key, speaker_label(len(speakers)))
        if not turns or turns[-1][0] != label:
            turns.append((label, []))
        turns[-1][1].append(str(text).strip())

    parts = [f"# {title}"]
    for label, texts in turns:
        parts.append(f"## {label}\n\n{' '.join(texts)}")
    return "\n\n".join(parts)


def combine_diarized_results(results: list[tuple[Any, float]]) -> dict[str, list[dict[str, Any]]]:
    """Offset timestamps and isolate anonymous speaker labels for each API request."""
    segments: list[dict[str, Any]] = []
    for part_number, (result, offset) in enumerate(results, start=1):
        part_segments = segment_value(result, "segments")
        if not part_segments:
            raise TranscriptionError("The API returned no diarized segments.")
        for segment in part_segments:
            speaker = segment_value(segment, "speaker")
            text = segment_value(segment, "text")
            start = segment_value(segment, "start")
            if speaker is None or text is None or start is None:
                raise TranscriptionError("The API returned an invalid diarized segment.")
            segments.append({
                "speaker": f"part_{part_number}:{speaker}",
                "start": float(start) + offset,
                "text": text,
            })
    return {"segments": segments}


def cleanup_prepared_audio(prepared_path: Path, workdir: Path) -> None:
    try:
        prepared_path.unlink(missing_ok=True)
        workdir.rmdir()
    except OSError:
        # Keep files if cleanup fails, or if another recording still has one.
        pass


def cleanup_diarization_parts(parts: list[tuple[Path, float]], workdir: Path) -> None:
    for path, _ in parts:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"Warning: could not remove temporary diarization part {path}: {exc}")
    try:
        workdir.rmdir()
    except OSError:
        pass


def transcribe_file(client: OpenAI, audio_path: Path, out_dir: Path, lang_mode: str, prompt: str | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lang = None if lang_mode == "auto" else lang_mode
    print(f"\n==> {audio_path.name} (lang={lang_mode})")

    prepared_path, is_temporary = prepare_audio_for_upload(audio_path, out_dir / "_prepared")
    if is_temporary:
        print(f"  Re-encoded oversized file for upload: {prepared_path.name}")
    parts, are_temporary = split_audio_for_diarization(prepared_path, out_dir / "_parts")
    if are_temporary:
        print(f"  Split long recording into {len(parts)} diarization part(s)")
    results = [(transcribe_audio_with_retry(client, path, lang), offset) for path, offset in parts]
    result = combine_diarized_results(results)
    transcript = format_diarized_transcript(result)
    readable_transcript = format_readable_transcript(result, audio_path.stem)
    out_path = out_dir / f"{audio_path.stem}.txt"
    readable_path = out_dir / "readable" / f"{audio_path.stem}.md"
    out_path.write_text(transcript, encoding="utf-8")
    readable_path.parent.mkdir(parents=True, exist_ok=True)
    readable_path.write_text(readable_transcript, encoding="utf-8")
    print(f"Saved: {out_path}")
    print(f"Saved: {readable_path}")
    if are_temporary:
        cleanup_diarization_parts(parts, (out_dir / "_parts"))
    if is_temporary:
        cleanup_prepared_audio(prepared_path, prepared_path.parent)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing audio files")
    parser.add_argument("--out", default="transcripts", help="Output folder (default: transcripts)")
    parser.add_argument("--lang", choices=["auto", "en", "fr"], default="auto",
                        help="Force language per file, or auto-detect (default: auto)")
    parser.add_argument("--prompt", default=None,
                        help="Deprecated compatibility option; ignored for diarization")
    args = parser.parse_args()
    if args.prompt:
        print("Warning: --prompt is ignored because the diarization model does not support it.")

    api_key = load_api_key()
    if not api_key:
        print(f"Missing OPENAI_API_KEY in {ENV_FILE}. Copy .env.example to .env and add your API key.")
        return 1

    in_dir = Path(args.folder).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    if not in_dir.is_dir():
        print(f"Not a folder: {in_dir}")
        return 1

    client = OpenAI(api_key=api_key, max_retries=0)
    audio_files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not audio_files:
        print(f"No audio files found in {in_dir} (supported: {sorted(SUPPORTED_EXTS)})")
        return 1

    failed_files: list[tuple[Path, Exception]] = []
    for audio_path in audio_files:
        try:
            transcribe_file(client, audio_path, out_dir, args.lang, args.prompt)
        except Exception as exc:
            print(f"Failed: {audio_path.name}: {exc}")
            failed_files.append((audio_path, exc))

    print(f"\nCompleted: {len(audio_files) - len(failed_files)}; Failed: {len(failed_files)}")
    for audio_path, error in failed_files:
        print(f"  - {audio_path.name}: {error}")
    return 1 if failed_files else 0


if __name__ == "__main__":
    sys.exit(main())
