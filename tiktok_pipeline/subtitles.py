from pathlib import Path

from tiktok_pipeline.models import ScriptDraft


# Phase 1A scaffold: write simple caption lines to preview formatting.
def create_subtitle_placeholder(script: ScriptDraft, renders_dir: Path) -> Path:
    lines = [f"{line.label}: {line.text}" for line in script.lines]
    out_path = renders_dir / "captions_placeholder.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
