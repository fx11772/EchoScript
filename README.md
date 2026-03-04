# ai_transcribe

Batch transcribe audio files in a folder using the OpenAI API.

## Usage
1. Install deps:
   - `pip install openai pydub`
   - `brew install ffmpeg`
2. Set env var:
   - `export OPENAI_API_KEY="..."`

Run:
```bash
python transcribe_folder.py ./audio_files --lang auto
