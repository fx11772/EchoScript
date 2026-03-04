import os
import sys
import argparse
from pathlib import Path
from openai import OpenAI
from pydub import AudioSegment

CHUNK_MINUTES = 10
SUPPORTED_EXTS = {".m4a", ".mp3", ".wav", ".aac", ".webm", ".mp4"}

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
        parts.append(transcribe_chunk(client, ch, lang, prompt))

    transcript = "\n\n".join(parts)
    out_path = out_dir / f"{audio_path.stem}.txt"
    out_path.write_text(transcript, encoding="utf-8")
    print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing audio files")
    parser.add_argument("--out", default="transcripts", help="Output folder (default: transcripts)")
    parser.add_argument("--lang", choices=["auto", "en", "fr"], default="auto",
                        help="Force language per file, or auto-detect (default: auto)")
    parser.add_argument("--prompt", default=None,
                        help="Optional context: names/acronyms to improve accuracy")
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
        transcribe_file(client, audio_path, out_dir, args.lang, args.prompt)

if __name__ == "__main__":
    main()