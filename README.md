
# EchoScript

EchoScript is now focused on **automating TikTok video generation** with a Python + FFmpeg CLI workflow.

Primary workflow:
1. Create a script from a specified niche.
2. Convert script to voice with timestamps (via forced alignment).
3. Select B-roll clips from disk using filename/tag matching against script keywords.
4. Assemble a catchy 9:16 TikTok video.
5. Burn subtitles onto the final video.

---

# Project Scope

In scope:
- End-to-end TikTok generation pipeline only.
- Python orchestration with FFmpeg CLI for media processing.
- Niche provided through CLI arguments in Phase 1.
- Forced alignment step for timestamp generation.
- B-roll selection via filename/tag keyword matching.
- Burned-in subtitles optimized for short-form content.

Out of scope:
- Any workflow outside the 5-step TikTok pipeline above.
- Non-CLI input/config systems for Phase 1.
- Replacing/removing the legacy transcription script.

---

# Subtitle Direction (TikTok)

Style rules:
- Short, punchy sentences (3-8 words each)
- High-impact, emotional wording
- Designed to stop the scroll in first 3 seconds
- Use curiosity, bold claims, or questions
- Avoid long explanations
- Maximum one idea per line

Required structure:
1. `HOOK` (shocking/intriguing)
2. `LINE` (2-4 lines building the idea)
3. `PAYOFF` (single clear takeaway)
4. `CTA` (comment/follow/share prompt)

Output format:
- Each subtitle line is separate.
- Example:
  - `HOOK: ...`
  - `LINE: ...`
  - `LINE: ...`
  - `PAYOFF: ...`
  - `CTA: ...`

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

# Current Entrypoints

- `transcribe_folder.py` (legacy side tool, kept as-is for transcription)
- `generate_tiktok.py` (Phase 1A scaffold entrypoint for TikTok pipeline)

Phase 1A scaffold example:

```bash
python generate_tiktok.py \
  --niche "fitness motivation" \
  --broll-dir ./broll \
  --out-dir ./output
```

---

# Legacy Transcription Tool (Side Tool)

The existing transcription utility remains available and is not removed.

Example transcription usage:

```bash
python transcribe_folder.py ./audio_files
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

# Implementation Plan

See:
- `IMPLEMENTATION_PLAN.txt`

---

# License

MIT License
