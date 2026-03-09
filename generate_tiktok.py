import argparse
import shutil
import sys
from pathlib import Path

from tiktok_pipeline import build_context, run_phase_1b
from tiktok_pipeline.utils import discover_broll_files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1A TikTok pipeline scaffolding")
    parser.add_argument("--niche", required=True, help="Niche/topic used to generate script")
    parser.add_argument("--broll-dir", required=True, help="Directory containing B-roll video clips")
    parser.add_argument("--out-dir", default="output", help="Output directory root (default: output)")
    parser.add_argument("--duration-seconds", type=int, default=30, help="Target duration in seconds")
    parser.add_argument("--voice", default=None, help="Voice id placeholder for future TTS step")
    parser.add_argument("--seed", type=int, default=None, help="Random seed placeholder")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temp artifacts")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run output for niche")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be a positive integer")

    broll_dir = Path(args.broll_dir).expanduser().resolve()
    if not broll_dir.is_dir():
        raise ValueError(f"B-roll directory not found: {broll_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    return broll_dir, out_dir


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        broll_dir, out_dir = validate_args(args)

        context = build_context(
            niche=args.niche,
            broll_dir=broll_dir,
            out_dir=out_dir,
            duration_seconds=args.duration_seconds,
            voice=args.voice,
            seed=args.seed,
            keep_temp=args.keep_temp,
            overwrite=args.overwrite,
        )

        if context.out_dir.exists() and args.overwrite:
            shutil.rmtree(context.out_dir)
            context = build_context(
                niche=args.niche,
                broll_dir=broll_dir,
                out_dir=out_dir,
                duration_seconds=args.duration_seconds,
                voice=args.voice,
                seed=args.seed,
                keep_temp=args.keep_temp,
                overwrite=args.overwrite,
            )

        broll_files = discover_broll_files(broll_dir)
        if not broll_files:
            print(f"No B-roll files found in {broll_dir}")
            return 1

        manifest_path = run_phase_1b(context)
        print("Phase 1B script and matching complete")
        print(f"Run directory: {context.out_dir}")
        print(f"Manifest: {manifest_path}")
        return 0
    except ValueError as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
