
# EchoScript

EchoScript is a lightweight Python tool that automatically transcribes long audio recordings into text using the OpenAI API.
It is designed for scenarios such as conference talks, lectures, interviews, or meetings where recordings can be **45–60 minutes or longer**.

The tool processes audio files in batches, splits large recordings into manageable chunks, sends them to the OpenAI transcription model, and reconstructs a clean transcript.

---

# Features

- Batch transcription of multiple audio files in a folder
- Automatic chunking for long recordings
- Supports common audio formats (`.m4a`, `.mp3`, `.wav`, `.aac`, `.webm`)
- Works well for long conference sessions
- Optional language forcing (`English`, `French`, or auto-detect)
- Clean text output per recording

---

# Use Case

This tool was created to transcribe technical conference talks (e.g., ConFoo sessions).
It allows processing many recordings automatically instead of manually uploading files one by one.

Example workflow:

1. Record conference talks using a phone or laptop.
2. Place recordings in a folder.
3. Run EchoScript.
4. Receive clean transcripts for each talk.

---

# Requirements

- Python **3.11–3.12**
- `ffmpeg`
- OpenAI API key

---

# Installation

Clone the repository:

```bash
git clone https://github.com/fx11772/EchoScript.git
cd EchoScript
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install openai pydub
```

Install ffmpeg (required for audio processing):

```bash
brew install ffmpeg
```

---

# Setup

Set your OpenAI API key as an environment variable.

macOS / Linux:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Windows (PowerShell):

```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

---

# Usage

Place your recordings in a folder.

Example structure:

```
audio_files/
  talk1.m4a
  talk2.m4a
  talk3.m4a
```

Run the transcription script:

```bash
python transcribe_folder.py ./audio_files
```

---

# Language Options

You can control the transcription language.

Auto detect (default):

```bash
python transcribe_folder.py ./audio_files --lang auto
```

Force English:

```bash
python transcribe_folder.py ./audio_files --lang en
```

Force French:

```bash
python transcribe_folder.py ./audio_files --lang fr
```

---

# Output

Transcripts are saved automatically in a `transcripts` folder.

Example:

```
transcripts/
  talk1.txt
  talk2.txt
  talk3.txt
```

Each file contains the full transcript of the original recording.

---

# Supported Audio Formats

- `.m4a`
- `.mp3`
- `.wav`
- `.aac`
- `.webm`
- `.mp4`

---

# How It Works

1. EchoScript scans the provided folder for audio files.
2. Each audio file is split into ~10 minute chunks.
3. Each chunk is sent to the OpenAI transcription model.
4. Transcriptions are merged into a final text file.

This approach avoids API size limits and improves reliability when processing long recordings.

---

# Example Command

```bash
python transcribe_folder.py ./conference_recordings --lang auto
```

---

# CI/CD (GitHub Actions)

This repository includes an automated CI/CD pipeline at:

`/.github/workflows/ci-cd.yml`

## Continuous Integration

On every pull request to `main` and every push to `main`, GitHub Actions:

1. Runs on Python `3.11` and `3.12`
2. Installs `ffmpeg`
3. Installs Python dependencies (`openai`, `pydub`)
4. Runs test suite:

```bash
python -m unittest discover -s tests -p 'test_*.py' -q
```

## Continuous Delivery

- Push to `main`: creates and uploads a build artifact (`.tar.gz`) in the workflow run.
- Push a version tag like `v1.0.0`: creates a GitHub Release and uploads the packaged bundle.

Create and push a release tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

---

# Possible Improvements

Future enhancements could include:

- Speaker detection
- Timestamped transcripts
- Automatic summaries of talks
- Markdown output formatting
- Keyword extraction
- CLI progress visualization

---

# License

MIT License
