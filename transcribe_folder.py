import sys
import argparse
import time
from pathlib import Path
from dotenv import dotenv_values
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from pydub import AudioSegment

CHUNK_MINUTES = 10
SUPPORTED_EXTS = {".m4a", ".mp3", ".wav", ".aac", ".webm", ".mp4"}
RETRY_DELAYS_SECONDS = (1, 2, 4)
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"


class ChunkTranscriptionError(RuntimeError):
    """Raised when a chunk cannot be transcribed after the allowed attempts."""


def load_api_key() -> str | None:
    """Return the OpenAI API key configured in the project .env file only."""
    api_key = dotenv_values(ENV_FILE).get("OPENAI_API_KEY")
    return api_key.strip() if isinstance(api_key, str) and api_key.strip() else None

def split_audio_to_chunks(audio_path: Path, workdir: Path) -> list[Path]:
    audio = AudioSegment.from_file(str(audio_path))
    chunk_ms = CHUNK_MINUTES * 60 * 1000

    chunks: list[Path] = []
    for i in range(0, len(audio), chunk_ms):
        chunk = audio[i:i + chunk_ms]
        chunk_path = workdir / f"{audio_path.stem}_chunk_{i//chunk_ms:03d}.m4a"
        chunk.export(str(chunk_path), format="mp4")  # m4a container (AAC), via ffmpeg
        chunks.append(chunk_path)

    return chunks

def transcribe_chunk(client: OpenAI, chunk_path: Path, lang: str | None, prompt: str | None) -> str:
    with open(chunk_path, "rb") as f:
        kwargs = {"model": "gpt-4o-mini-transcribe", "file": f}
        if lang:
            kwargs["language"] = lang
        if prompt:
            kwargs["prompt"] = prompt
        result = client.audio.transcriptions.create(**kwargs)
    return result.text


def is_retryable_error(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True

    return isinstance(error, APIStatusError) and 500 <= error.status_code < 600


def transcribe_chunk_with_retry(
    client: OpenAI,
    chunk_path: Path,
    lang: str | None,
    prompt: str | None,
    chunk_number: int,
    total_chunks: int,
) -> str:
    for attempt, delay in enumerate((*RETRY_DELAYS_SECONDS, None), start=1):
        try:
            return transcribe_chunk(client, chunk_path, lang, prompt)
        except Exception as exc:
            if not is_retryable_error(exc) or delay is None:
                raise ChunkTranscriptionError(
                    f"Chunk {chunk_number}/{total_chunks} ({chunk_path.name}) failed: {exc}"
                ) from exc

            print(
                f"  Transient error on chunk {chunk_number}/{total_chunks}: {exc}. "
                f"Retrying in {delay} second(s) ({attempt}/{len(RETRY_DELAYS_SECONDS)})..."
            )
            time.sleep(delay)


def cleanup_chunks(chunks: list[Path], tmp_dir: Path) -> None:
    removed_chunks = 0
    for chunk_path in chunks:
        try:
            existed = chunk_path.exists()
            chunk_path.unlink(missing_ok=True)
            removed_chunks += existed
        except OSError as exc:
            print(f"Warning: could not remove temporary chunk {chunk_path}: {exc}")

    print(f"Removed {removed_chunks} temporary chunk(s)")

    try:
        tmp_dir.rmdir()
        print("Removed temporary chunks directory")
    except OSError:
        # The directory is retained when it contains chunks from another file or
        # when a chunk could not be removed.
        pass


def transcribe_file(client: OpenAI, audio_path: Path, out_dir: Path, lang_mode: str, prompt: str | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_chunks"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    lang = None if lang_mode == "auto" else lang_mode

    print(f"\n==> {audio_path.name} (lang={lang_mode})")
    chunks = split_audio_to_chunks(audio_path, tmp_dir)
    print(f"Chunks: {len(chunks)} (~{CHUNK_MINUTES} min each)")

    parts: list[str] = []
    for idx, ch in enumerate(chunks, start=1):
        print(f"  Transcribing chunk {idx}/{len(chunks)}")
        parts.append(transcribe_chunk_with_retry(client, ch, lang, prompt, idx, len(chunks)))

    transcript = "\n\n".join(parts)
    out_path = out_dir / f"{audio_path.stem}.txt"
    out_path.write_text(transcript, encoding="utf-8")
    print(f"Saved: {out_path}")
    cleanup_chunks(chunks, tmp_dir)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing audio files")
    parser.add_argument("--out", default="transcripts", help="Output folder (default: transcripts)")
    parser.add_argument("--lang", choices=["auto", "en", "fr"], default="auto",
                        help="Force language per file, or auto-detect (default: auto)")
    parser.add_argument("--prompt", default=None,
                        help="Optional context: names/acronyms to improve accuracy")
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        print(
            f"Missing OPENAI_API_KEY in {ENV_FILE}. "
            "Copy .env.example to .env and add your API key."
        )
        return 1

    in_dir = Path(args.folder).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not in_dir.is_dir():
        print(f"Not a folder: {in_dir}")
        return 1

    # Retries are handled explicitly per chunk so attempts and delays remain
    # predictable and visible to the user.
    client = OpenAI(api_key=api_key, max_retries=0)

    audio_files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
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

    completed_files = len(audio_files) - len(failed_files)
    print(f"\nCompleted: {completed_files}; Failed: {len(failed_files)}")
    for audio_path, error in failed_files:
        print(f"  - {audio_path.name}: {error}")

    return 1 if failed_files else 0

if __name__ == "__main__":
    sys.exit(main())
