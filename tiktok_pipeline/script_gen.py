import os
import random
import re

from openai import OpenAI

from tiktok_pipeline.models import ScriptDraft, ScriptLine

MIN_WORDS = 3
MAX_WORDS = 8
ALLOWED_LABELS = {"HOOK", "LINE", "PAYOFF", "CTA"}
STOP_WORDS = {"the", "a", "an", "of", "for", "to", "and", "in", "on", "with"}
SCRIPT_MODEL = "gpt-4o-mini"
MAX_SCRIPT_ATTEMPTS = 3
CREATIVE_ANGLES = [
    "old money mystique",
    "silent power and status",
    "private jet ambition",
    "elite routine psychology",
    "wealth signaling through restraint",
    "high-status discipline",
    "luxury as identity transformation",
    "scarcity and exclusivity tension",
]
HOOK_PATTERNS = [
    "start with a provocative question",
    "start with a bold forbidden truth",
    "start with a status-shifting claim",
    "start with a curiosity gap",
    "start with a dangerous-sounding insight",
]
FRESHNESS_RULES = [
    "Avoid generic guru cliches.",
    "Do not repeat common TikTok wording.",
    "Do not use 'here is the secret' phrasing.",
    "Favor specificity over vague hype.",
]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text)


def parse_script_response(niche: str, raw_text: str) -> ScriptDraft:
    lines: list[ScriptLine] = []
    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise ValueError(f"Invalid script line format: {stripped}")
        label, text = stripped.split(":", 1)
        lines.append(ScriptLine(label=label.strip().upper(), text=text.strip()))

    script = ScriptDraft(niche=niche, lines=lines)
    validate_script_format(script)
    return script


def _variation_brief(niche: str, seed: int | None = None) -> str:
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    angle = rng.choice(CREATIVE_ANGLES)
    hook_pattern = rng.choice(HOOK_PATTERNS)
    freshness = rng.choice(FRESHNESS_RULES)
    return (
        f"Niche: {niche}. "
        f"Creative angle: {angle}. "
        f"Hook instruction: {hook_pattern}. "
        f"{freshness}"
    )


def _build_messages(niche: str, seed: int | None = None) -> list[dict[str, str]]:
    variation_brief = _variation_brief(niche, seed=seed)
    return [
        {
            "role": "system",
            "content": (
                "You write short-form TikTok scripts. "
                "Follow these rules exactly: short punchy lines, 3-8 words each, one idea per line, "
                "dramatic, motivational, slightly provocative tone. "
                "Make the output feel fresh and non-repetitive. "
                "Return only separate lines in this format: "
                "HOOK: ..., LINE: ..., LINE: ..., PAYOFF: ..., CTA: ..."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Create a TikTok script for the niche '{niche}'. "
                "Structure must be: 1 HOOK, 2 to 4 LINE entries, 1 PAYOFF, 1 CTA. "
                f"{variation_brief}"
            ),
        },
    ]


def validate_script_format(script: ScriptDraft) -> None:
    if len(script.lines) < 5:
        raise ValueError("Script must include at least 5 lines")

    labels = [line.label for line in script.lines]
    if labels[0] != "HOOK":
        raise ValueError("First line must be HOOK")
    if labels[-2] != "PAYOFF":
        raise ValueError("Second to last line must be PAYOFF")
    if labels[-1] != "CTA":
        raise ValueError("Last line must be CTA")

    line_count = labels.count("LINE")
    if line_count < 2 or line_count > 4:
        raise ValueError("Script must include 2-4 LINE entries")

    for line in script.lines:
        if line.label not in ALLOWED_LABELS:
            raise ValueError(f"Unsupported label: {line.label}")
        words = _words(line.text)
        if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
            raise ValueError(
                f"Line '{line.label}: {line.text}' must have {MIN_WORDS}-{MAX_WORDS} words"
            )


def generate_script(
    niche: str,
    client: OpenAI | None = None,
    model: str = SCRIPT_MODEL,
    seed: int | None = None,
) -> ScriptDraft:
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable.")
        client = OpenAI(api_key=api_key)

    last_error: Exception | None = None
    for _attempt in range(MAX_SCRIPT_ATTEMPTS):
        response = client.chat.completions.create(
            model=model,
            messages=_build_messages(niche, seed=seed),
            temperature=1.3,
        )
        content = response.choices[0].message.content or ""
        try:
            return parse_script_response(niche, content)
        except ValueError as exc:
            last_error = exc

    raise ValueError(
        f"OpenAI did not return a valid script after {MAX_SCRIPT_ATTEMPTS} attempts"
    ) from last_error
