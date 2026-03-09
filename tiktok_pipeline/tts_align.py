import os
import re
from pathlib import Path

from openai import OpenAI

from tiktok_pipeline.models import ScriptDraft, TimedLine, TimedToken
from tiktok_pipeline.utils import write_json

TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
ALIGNMENT_MODEL = "whisper-1"


def script_text(script: ScriptDraft) -> str:
    return " ".join(line.text.strip() for line in script.lines)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def _read_transcription_text(response) -> str:
    if hasattr(response, "text") and response.text:
        return response.text
    if isinstance(response, dict):
        return response.get("text", "")
    return ""


def _read_transcription_duration(response) -> float | None:
    if hasattr(response, "duration") and response.duration is not None:
        return float(response.duration)
    if isinstance(response, dict) and response.get("duration") is not None:
        return float(response["duration"])
    return None


def _read_transcription_words(response) -> list[TimedToken]:
    raw_words = None
    if hasattr(response, "words") and response.words is not None:
        raw_words = response.words
    elif isinstance(response, dict):
        raw_words = response.get("words")

    if not raw_words:
        return []

    tokens: list[TimedToken] = []
    for item in raw_words:
        if isinstance(item, dict):
            word = item.get("word", "")
            start = item.get("start", 0.0)
            end = item.get("end", start)
        else:
            word = getattr(item, "word", "")
            start = getattr(item, "start", 0.0)
            end = getattr(item, "end", start)
        tokens.append(TimedToken(token=str(word).strip(), start_s=float(start), end_s=float(end)))
    return [token for token in tokens if token.token]


def _estimate_word_timestamps(text: str, duration_s: float) -> list[TimedToken]:
    words = _words(text)
    if not words:
        return []

    slice_duration = duration_s / len(words) if duration_s > 0 else 0.0
    tokens: list[TimedToken] = []
    for idx, word in enumerate(words):
        start_s = round(idx * slice_duration, 3)
        end_s = round((idx + 1) * slice_duration, 3)
        tokens.append(TimedToken(token=word, start_s=start_s, end_s=end_s))
    return tokens


def normalize_timed_lines(script: ScriptDraft, timed_tokens: list[TimedToken], duration_s: float) -> list[TimedLine]:
    script_word_counts = [len(_words(line.text)) for line in script.lines]
    if not timed_tokens:
        timed_tokens = _estimate_word_timestamps(script_text(script), duration_s)

    if not timed_tokens:
        return [
            TimedLine(label=line.label, text=line.text, start_s=0.0, end_s=0.0, tokens=[])
            for line in script.lines
        ]

    total_script_words = max(sum(script_word_counts), 1)
    total_timed_words = len(timed_tokens)
    allocated = 0
    cursor = 0
    timed_lines: list[TimedLine] = []

    for idx, line in enumerate(script.lines):
        remaining_lines = len(script.lines) - idx
        remaining_tokens = total_timed_words - cursor
        target = round((script_word_counts[idx] / total_script_words) * total_timed_words)
        target = max(target, 1 if remaining_tokens > 0 else 0)
        max_for_line = remaining_tokens - max(remaining_lines - 1, 0)
        if max_for_line < 0:
            max_for_line = 0
        if remaining_lines == 1:
            count = remaining_tokens
        else:
            count = min(target, max_for_line)
        line_tokens = timed_tokens[cursor : cursor + count]
        cursor += count
        allocated += count

        if line_tokens:
            start_s = line_tokens[0].start_s
            end_s = line_tokens[-1].end_s
        else:
            anchor = timed_lines[-1].end_s if timed_lines else 0.0
            start_s = anchor
            end_s = anchor

        timed_lines.append(
            TimedLine(
                label=line.label,
                text=line.text,
                start_s=round(start_s, 3),
                end_s=round(end_s, 3),
                tokens=line_tokens,
            )
        )

    if allocated < total_timed_words and timed_lines:
        trailing = timed_tokens[allocated:]
        timed_lines[-1].tokens.extend(trailing)
        timed_lines[-1].end_s = trailing[-1].end_s

    return timed_lines


def _timed_lines_to_dict(timed_lines: list[TimedLine]) -> list[dict]:
    return [
        {
            "label": line.label,
            "text": line.text,
            "start_s": line.start_s,
            "end_s": line.end_s,
            "tokens": [
                {"token": token.token, "start_s": token.start_s, "end_s": token.end_s}
                for token in line.tokens
            ],
        }
        for line in timed_lines
    ]


def create_narration_and_alignment(
    script: ScriptDraft,
    audio_dir: Path,
    alignments_dir: Path,
    voice: str | None = None,
    client: OpenAI | None = None,
) -> tuple[Path, Path]:
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable.")
        client = OpenAI(api_key=api_key)

    narration_path = audio_dir / "narration.mp3"
    alignment_path = alignments_dir / "alignment.json"
    voice_id = voice or TTS_VOICE

    speech_response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=voice_id,
        input=script_text(script),
    )
    speech_response.stream_to_file(str(narration_path))

    with open(narration_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=ALIGNMENT_MODEL,
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

    raw_text = _read_transcription_text(transcription)
    duration_s = _read_transcription_duration(transcription) or 0.0
    timed_tokens = _read_transcription_words(transcription)
    timed_lines = normalize_timed_lines(script, timed_tokens, duration_s)

    payload = {
        "niche": script.niche,
        "voice": voice_id,
        "tts_model": TTS_MODEL,
        "alignment_model": ALIGNMENT_MODEL,
        "transcript_text": raw_text,
        "duration_s": duration_s,
        "timed_lines": _timed_lines_to_dict(timed_lines),
    }
    write_json(alignment_path, payload)
    return narration_path, alignment_path
