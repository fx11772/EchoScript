import random
import re
from pathlib import Path

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


def _folder_key(file_path: Path) -> str:
    parent = file_path.parent
    return str(parent)


def _choose_candidate_from_folder(
    folder_candidates: list[tuple[tuple[float, float, float], float, Path]]
) -> tuple[tuple[float, float, float], float, Path]:
    folder_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return folder_candidates[0]


def extract_keywords(script: ScriptDraft) -> set[str]:
    keywords: set[str] = set()
    for line in script.lines:
        keywords.update(_tokens(line.text))
    return keywords


def score_broll_file(file_path: Path, keywords: set[str]) -> tuple[int, set[str]]:
    file_terms = _tokens(file_path.stem)
    matched = file_terms.intersection(keywords)
    return (len(matched), matched)


def build_clip_plan(script: ScriptDraft, broll_files: list[Path], seed: int | None = None) -> list[ClipPlanItem]:
    if not broll_files:
        return []

    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    items: list[ClipPlanItem] = []
    folders: dict[str, list[Path]] = {}
    for candidate in broll_files:
        folders.setdefault(_folder_key(candidate), []).append(candidate)

    folder_cycle: list[str] = []
    previous_folder: str | None = None

    for idx, line in enumerate(script.lines):
        if not folder_cycle:
            folder_cycle = list(folders.keys())
            rng.shuffle(folder_cycle)
            if previous_folder is not None and len(folder_cycle) > 1 and folder_cycle[0] == previous_folder:
                folder_cycle.append(folder_cycle.pop(0))

        selected_folder = folder_cycle.pop(0)
        folder_files = list(folders[selected_folder])
        rng.shuffle(folder_files)
        best_file = folder_files[0]
        previous_folder = selected_folder
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
