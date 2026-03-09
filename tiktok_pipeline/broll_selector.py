from pathlib import Path

from tiktok_pipeline.models import ClipPlanItem, ScriptDraft


# Phase 1A scaffold: assign clips in order as placeholder selection strategy.
def build_clip_plan_stub(script: ScriptDraft, broll_files: list[Path]) -> list[ClipPlanItem]:
    if not broll_files:
        return []

    items: list[ClipPlanItem] = []
    for idx, line in enumerate(script.lines):
        clip_path = broll_files[idx % len(broll_files)]
        start_s = float(idx * 2)
        end_s = start_s + 2.0
        items.append(
            ClipPlanItem(
                line_label=line.label,
                line_text=line.text,
                clip_path=clip_path,
                start_s=start_s,
                end_s=end_s,
            )
        )
    return items
