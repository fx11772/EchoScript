# EchoScript

EchoScript transcribes a folder of recordings with OpenAI's native speaker
diarization. Each recording produces a timestamped text file with consistent,
anonymous speaker labels for that recording.

## Features

- Batch processing for `.m4a`, `.mp3`, `.wav`, `.aac`, `.webm`, and `.mp4`
- Native OpenAI diarization with `gpt-4o-transcribe-diarize`
- Timestamped output such as `[01:05] Speaker B: ...`
- Optional language forcing (`en`, `fr`, or automatic detection)
- Retries for temporary API and network failures; a failed file does not stop
  the rest of the batch

## Requirements

- Python 3.11–3.12
- `ffmpeg` (and its bundled `ffprobe`) for oversized recordings
- An OpenAI API key

Install dependencies and ffmpeg on macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

Create `.env` from the example and add the key:

```dotenv
OPENAI_API_KEY="your_api_key_here"
```

EchoScript reads the key only from the project's `.env` file. Do not commit or
share that file.

## Usage

```bash
python transcribe_folder.py ./audio_files
python transcribe_folder.py ./audio_files --lang fr
python transcribe_folder.py ./audio_files --out ./transcripts
```

`--lang auto` is the default. `--prompt` remains accepted for compatibility,
but is ignored with a warning because OpenAI's diarization model does not
support prompts.

## Output

Files are written to `transcripts/<recording-name>.txt` by default. A file
looks like this:

```text
[00:00] Speaker A: Bonjour, on commence la réunion.
[00:02] Je vais présenter le premier point.

[00:04] Speaker B: Parfait, merci.
[01:05] Speaker A: Premier point à l'ordre du jour.
```

The labels are deliberately anonymous. `Speaker A`, `Speaker B`, and so on
are consistent within a single uploaded recording, but are not identities and
are not guaranteed to refer to the same person in another file.

## Large recordings

Every recording is sent in one request, which lets the diarization model keep
speaker labels consistent across the whole file. Files at or below the OpenAI
audio upload limit (25 MiB) are uploaded unchanged. Larger files are
temporarily re-encoded by `ffmpeg` as mono 16 kHz AAC at a bitrate calculated
to target the limit. If the prepared file is still too large, EchoScript fails
that recording with an explicit error and continues with the remaining files.

Temporary prepared audio is removed after a successful transcription. It is
left in `transcripts/_prepared` on failure for troubleshooting.

## Failure behavior

Temporary OpenAI API or network failures are retried three times with
increasing delays. Invalid requests, authentication errors, audio-preparation
errors, and filesystem errors fail immediately. The command reports all failed
recordings and exits with a non-zero status if any recording failed.

## License

MIT License
