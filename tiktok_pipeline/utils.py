import json
import re
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}


def slugify(value: str) -> str:
    lowered = value.lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", "-", lowered)
    compact = re.sub(r"-+", "-", normalized).strip("-")
    return compact or "niche"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_broll_files(broll_dir: Path) -> list[Path]:
    files = [p for p in broll_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(files)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
