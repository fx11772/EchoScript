#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiktok_pipeline.pexels import (
    build_download_candidates,
    build_download_filename,
    download_file,
    get_api_key,
    search_videos,
)
from tiktok_pipeline.utils import ensure_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download portrait B-roll clips from Pexels into a local folder."
    )
    parser.add_argument("--niche", required=True, help="Primary niche phrase to search for.")
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Optional search query override. Repeat to broaden the pull.",
    )
    parser.add_argument("--out-dir", default="./broll", help="Destination folder for downloaded clips.")
    parser.add_argument(
        "--count",
        type=int,
        default=12,
        help="Maximum number of unique clips to download.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=15,
        help="Number of clips to request per API page.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=3,
        help="Maximum pages to scan per query.",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=3,
        help="Minimum video duration in seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=20,
        help="Maximum video duration in seconds.",
    )
    parser.add_argument(
        "--api-key-env",
        default="PEXELS_API_KEY",
        help="Environment variable containing the Pexels API key.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching clips without downloading them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.count <= 0:
        raise ValueError("--count must be greater than 0")
    if args.per_page <= 0 or args.pages <= 0:
        raise ValueError("--per-page and --pages must be greater than 0")
    if args.min_duration < 0:
        raise ValueError("--min-duration must be 0 or greater")
    if args.max_duration < args.min_duration:
        raise ValueError("--max-duration must be greater than or equal to --min-duration")

    api_key = get_api_key(args.api_key_env)
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    queries = args.query or [args.niche]

    manifest_path = out_dir / "pexels_manifest.json"
    existing_manifest = load_existing_manifest(manifest_path)
    seen_video_ids: set[int] = {
        int(item["video_id"])
        for item in existing_manifest
        if isinstance(item, dict) and "video_id" in item
    }
    manifest_entries: list[dict[str, object]] = list(existing_manifest)
    target_total = len(manifest_entries) + args.count

    for query in queries:
        if len(manifest_entries) >= target_total:
            break
        for page in range(1, args.pages + 1):
            if len(manifest_entries) >= target_total:
                break

            payload = search_videos(
                api_key,
                query,
                page=page,
                per_page=args.per_page,
                orientation="portrait",
            )
            candidates = build_download_candidates(
                payload,
                query,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
            )
            if not candidates:
                continue

            for candidate in candidates:
                if candidate.video_id in seen_video_ids:
                    continue
                seen_video_ids.add(candidate.video_id)

                filename = build_download_filename(
                    args.niche,
                    candidate.source_query,
                    candidate.video_id,
                    candidate.file_type,
                )
                dest_path = out_dir / filename
                if not args.dry_run:
                    download_file(api_key, candidate.download_url, dest_path)

                manifest_entries.append(
                    {
                        "video_id": candidate.video_id,
                        "query": candidate.source_query,
                        "duration": candidate.duration,
                        "file_type": candidate.file_type,
                        "width": candidate.width,
                        "height": candidate.height,
                        "pexels_url": candidate.pexels_url,
                        "download_url": candidate.download_url,
                        "path": str(dest_path),
                    }
                )
                if len(manifest_entries) >= target_total:
                    break

    manifest_path.write_text(
        json.dumps(
            {
                "niche": args.niche,
                "queries": queries,
                "count": len(manifest_entries),
                "dry_run": args.dry_run,
                "clips": manifest_entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Prepared {len(manifest_entries) - len(existing_manifest)} new clip(s) in {out_dir}")
    print(f"Total clips tracked: {len(manifest_entries)}")
    print(f"Manifest: {manifest_path}")
    return 0


def load_existing_manifest(manifest_path: Path) -> list[dict[str, object]]:
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = payload.get("clips", [])
    return [item for item in clips if isinstance(item, dict)]


if __name__ == "__main__":
    raise SystemExit(main())
