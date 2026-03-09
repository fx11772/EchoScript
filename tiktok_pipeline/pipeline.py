from pathlib import Path

from tiktok_pipeline.assemble import create_assembly_plan_placeholder
from tiktok_pipeline.broll_selector import build_clip_plan_stub
from tiktok_pipeline.models import PipelineContext
from tiktok_pipeline.script_gen import generate_script_stub
from tiktok_pipeline.subtitles import create_subtitle_placeholder
from tiktok_pipeline.tts_align import create_alignment_placeholder
from tiktok_pipeline.utils import discover_broll_files, ensure_dir, slugify, write_json


def build_context(
    niche: str,
    broll_dir: Path,
    out_dir: Path,
    duration_seconds: int,
    voice: str | None,
    seed: int | None,
    keep_temp: bool,
    overwrite: bool,
) -> PipelineContext:
    run_slug = slugify(niche)
    run_root = out_dir / run_slug

    scripts_dir = ensure_dir(run_root / "scripts")
    audio_dir = ensure_dir(run_root / "audio")
    alignments_dir = ensure_dir(run_root / "alignments")
    plans_dir = ensure_dir(run_root / "plans")
    temp_dir = ensure_dir(run_root / "temp")
    renders_dir = ensure_dir(run_root / "renders")

    return PipelineContext(
        niche=niche,
        broll_dir=broll_dir,
        out_dir=run_root,
        scripts_dir=scripts_dir,
        audio_dir=audio_dir,
        alignments_dir=alignments_dir,
        plans_dir=plans_dir,
        temp_dir=temp_dir,
        renders_dir=renders_dir,
        duration_seconds=duration_seconds,
        voice=voice,
        seed=seed,
        keep_temp=keep_temp,
        overwrite=overwrite,
    )


def run_phase_1a(context: PipelineContext) -> Path:
    broll_files = discover_broll_files(context.broll_dir)
    script = generate_script_stub(context.niche)
    clip_plan = build_clip_plan_stub(script, broll_files)
    alignment_path = create_alignment_placeholder(script, context.alignments_dir)
    assembly_plan_path = create_assembly_plan_placeholder(context.plans_dir, clip_plan)
    captions_path = create_subtitle_placeholder(script, context.renders_dir)

    script_path = context.scripts_dir / "script_draft.json"
    write_json(script_path, script.to_dict())

    manifest = {
        "phase": "1A",
        "status": "scaffolded",
        "niche": context.niche,
        "broll_dir": str(context.broll_dir),
        "broll_count": len(broll_files),
        "duration_seconds": context.duration_seconds,
        "voice": context.voice,
        "seed": context.seed,
        "keep_temp": context.keep_temp,
        "overwrite": context.overwrite,
        "artifacts": {
            "script": str(script_path),
            "alignment_placeholder": str(alignment_path),
            "assembly_plan_placeholder": str(assembly_plan_path),
            "captions_placeholder": str(captions_path),
        },
    }
    manifest_path = context.out_dir / "phase_1a_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path
