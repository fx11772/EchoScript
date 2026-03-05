import os
import sys
import argparse
import time
from pathlib import Path
from openai import OpenAI
from pydub import AudioSegment

CHUNK_MINUTES = 10
SUPPORTED_EXTS = {".m4a", ".mp3", ".wav", ".aac", ".webm", ".mp4"}
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.0

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
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            with open(chunk_path, "rb") as f:
                kwargs = {"model": "gpt-4o-mini-transcribe", "file": f}
                if lang:
                    kwargs["language"] = lang
                if prompt:
                    kwargs["prompt"] = prompt
                result = client.audio.transcriptions.create(**kwargs)
            return result.text
        except Exception as exc:
            if attempt == MAX_RETRY_ATTEMPTS:
                raise RuntimeError(
                    f"Failed to transcribe chunk {chunk_path.name} after {MAX_RETRY_ATTEMPTS} attempts"
                ) from exc
            delay = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"  Retry {attempt}/{MAX_RETRY_ATTEMPTS - 1} in {delay:.1f}s after error: {exc}")
            time.sleep(delay)
    raise RuntimeError(f"Unexpected retry flow for chunk {chunk_path.name}")

def transcribe_file(
    client: OpenAI,
    audio_path: Path,
    out_dir: Path,
    lang_mode: str,
    prompt: str | None,
    cleanup_chunks: bool = True,
    overwrite: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{audio_path.stem}.txt"
    if out_path.exists() and not overwrite:
        print(f"\n==> {audio_path.name} (lang={lang_mode})")
        print(f"Skipping: {out_path} already exists (use --overwrite to regenerate)")
        return

    tmp_dir = out_dir / "_chunks"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    lang = None if lang_mode == "auto" else lang_mode

    print(f"\n==> {audio_path.name} (lang={lang_mode})")
    chunks = split_audio_to_chunks(audio_path, tmp_dir)
    print(f"Chunks: {len(chunks)} (~{CHUNK_MINUTES} min each)")

    parts: list[str] = []
    try:
        for idx, ch in enumerate(chunks, start=1):
            print(f"  Transcribing chunk {idx}/{len(chunks)}")
            parts.append(transcribe_chunk(client, ch, lang, prompt))

        transcript = "\n\n".join(parts)
        out_path.write_text(transcript, encoding="utf-8")
        print(f"Saved: {out_path}")
    finally:
        if cleanup_chunks:
            for ch in chunks:
                ch.unlink(missing_ok=True)
            try:
                tmp_dir.rmdir()
            except OSError:
                pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing audio files")
    parser.add_argument("--out", default="transcripts", help="Output folder (default: transcripts)")
    parser.add_argument("--lang", choices=["auto", "en", "fr"], default="auto",
                        help="Force language per file, or auto-detect (default: auto)")
    parser.add_argument("--prompt", default=None,
                        help="Optional context: names/acronyms to improve accuracy")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing transcript files (default: skip existing)")
    parser.add_argument("--keep-chunks", dest="cleanup_chunks", action="store_false",
                        help="Keep temporary chunk files in the output _chunks folder")
    parser.set_defaults(cleanup_chunks=True)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY environment variable.")
        sys.exit(1)

    in_dir = Path(args.folder).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not in_dir.is_dir():
        print(f"Not a folder: {in_dir}")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    audio_files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
    if not audio_files:
        print(f"No audio files found in {in_dir} (supported: {sorted(SUPPORTED_EXTS)})")
        sys.exit(1)

    for audio_path in audio_files:
        transcribe_file(
            client=client,
            audio_path=audio_path,
            out_dir=out_dir,
            lang_mode=args.lang,
            prompt=args.prompt,
            cleanup_chunks=args.cleanup_chunks,
            overwrite=args.overwrite,
        )

if __name__ == "__main__":
    main()
