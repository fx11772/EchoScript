from pathlib import Path

from tiktok_pipeline.models import ClipPlanItem
from tiktok_pipeline.utils import write_json


# Phase 1A scaffold: write FFmpeg assembly plan placeholder.
def create_assembly_plan_placeholder(plans_dir: Path, clip_plan: list[ClipPlanItem]) -> Path:
    payload = {
        "status": "placeholder",
        "step": "assemble",
        "note": "FFmpeg assembly implementation scheduled for Phase 1D",
        "clip_plan": [
            {
                "line_label": item.line_label,
                "line_text": item.line_text,
                "clip_path": str(item.clip_path) if item.clip_path else None,
                "start_s": item.start_s,
                "end_s": item.end_s,
            }
            for item in clip_plan
        ],
    }
    out_path = plans_dir / "assembly_plan_placeholder.json"
    write_json(out_path, payload)
    return out_path
