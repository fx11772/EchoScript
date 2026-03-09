from pathlib import Path

from tiktok_pipeline.models import ScriptDraft
from tiktok_pipeline.utils import write_json


# Phase 1A scaffold: create placeholder alignment artifact.
def create_alignment_placeholder(script: ScriptDraft, alignments_dir: Path) -> Path:
    payload = {
        "niche": script.niche,
        "status": "placeholder",
        "line_count": len(script.lines),
        "note": "Forced alignment to be implemented in Phase 1C",
    }
    out_path = alignments_dir / "alignment_placeholder.json"
    write_json(out_path, payload)
    return out_path
