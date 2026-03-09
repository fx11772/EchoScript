from pathlib import Path
import re

from tiktok_pipeline.models import ClipPlanItem, ScriptDraft

STOP_WORDS = {
    "the",
    "a",
    "an",
    "of",
    "for",
    "to",
    "and",
    "in",
    "on",
    "with",
    "if",
    "what",
    "this",
    "that",
}


def _tokens(value: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9']+", value.lower())
    return {w for w in words if len(w) > 2 and w not in STOP_WORDS}


def _parent_tokens(file_path: Path) -> set[str]:
    tokens: set[str] = set()
    for part in file_path.parts[:-1]:
        tokens.update(_tokens(part))
    return tokens


def extract_keywords(script: ScriptDraft) -> set[str]:
    keywords: set[str] = set()
    for line in script.lines:
        keywords.update(_tokens(line.text))
    return keywords


def score_broll_file(file_path: Path, keywords: set[str]) -> tuple[int, set[str]]:
    file_terms = _tokens(file_path.stem)
    matched = file_terms.intersection(keywords)
    return (len(matched), matched)


def build_clip_plan(script: ScriptDraft, broll_files: list[Path]) -> list[ClipPlanItem]:
    if not broll_files:
        return []

    script_keywords = extract_keywords(script)
    niche_keywords = _tokens(script.niche)
    items: list[ClipPlanItem] = []
    fallback_idx = 0

    for idx, line in enumerate(script.lines):
        line_keywords = _tokens(line.text)
        best_file = None
        best_rank = (-1, -1, "")
        for candidate in broll_files:
            score, _matched = score_broll_file(candidate, line_keywords or script_keywords)
            niche_folder_score = len(_parent_tokens(candidate).intersection(niche_keywords))
            rank = (score, niche_folder_score, str(candidate))
            if rank > best_rank:
                best_rank = rank
                best_file = candidate

        if best_file is None or best_rank[0] == 0:
            best_file = broll_files[fallback_idx % len(broll_files)]
            fallback_idx += 1

        start_s = float(idx * 2)
        end_s = start_s + 2.0
        items.append(
            ClipPlanItem(
                line_label=line.label,
                line_text=line.text,
                clip_path=best_file,
                start_s=start_s,
                end_s=end_s,
            )
        )
    return items
