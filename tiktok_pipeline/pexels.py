import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from tiktok_pipeline.utils import ensure_dir, slugify

PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"
TARGET_ASPECT_RATIO = 9 / 16


@dataclass(frozen=True)
class DownloadCandidate:
    video_id: int
    source_query: str
    duration: int
    pexels_url: str
    download_url: str
    width: int
    height: int
    file_type: str


def get_api_key(env_var: str = "PEXELS_API_KEY") -> str:
    api_key = os.getenv(env_var, "").strip()
    if not api_key:
        raise ValueError(f"Missing Pexels API key in environment variable: {env_var}")
    return api_key


def search_videos(
    api_key: str,
    query: str,
    *,
    page: int = 1,
    per_page: int = 15,
    orientation: str = "portrait",
    size: str = "medium",
    locale: str = "en-US",
) -> dict[str, Any]:
    params = urlencode(
        {
            "query": query,
            "page": page,
            "per_page": per_page,
            "orientation": orientation,
            "size": size,
            "locale": locale,
        }
    )
    req = Request(
        f"{PEXELS_VIDEO_SEARCH_URL}?{params}",
        headers={"Authorization": api_key, "User-Agent": "EchoScript/1.0"},
    )
    with urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def build_download_candidates(
    payload: dict[str, Any],
    source_query: str,
    *,
    min_duration: int = 0,
    max_duration: int | None = None,
) -> list[DownloadCandidate]:
    candidates: list[DownloadCandidate] = []
    for video in payload.get("videos", []):
        duration = int(video.get("duration") or 0)
        if duration < min_duration:
            continue
        if max_duration is not None and duration > max_duration:
            continue

        chosen = select_best_video_file(video.get("video_files", []))
        if chosen is None:
            continue

        candidates.append(
            DownloadCandidate(
                video_id=int(video["id"]),
                source_query=source_query,
                duration=duration,
                pexels_url=str(video.get("url") or ""),
                download_url=str(chosen["link"]),
                width=int(chosen.get("width") or 0),
                height=int(chosen.get("height") or 0),
                file_type=str(chosen.get("file_type") or "video/mp4"),
            )
        )
    return candidates


def select_best_video_file(video_files: list[dict[str, Any]]) -> dict[str, Any] | None:
    viable = [item for item in video_files if item.get("link")]
    if not viable:
        return None

    def sort_key(item: dict[str, Any]) -> tuple[int, int, float, int]:
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        ratio = (width / height) if width and height else 999.0
        is_mp4 = 0 if str(item.get("file_type", "")).lower() == "video/mp4" else 1
        is_portrait = 0 if height > width else 1
        ratio_delta = abs(ratio - TARGET_ASPECT_RATIO)
        area_penalty = -(width * height)
        return (is_mp4, is_portrait, ratio_delta, area_penalty)

    return min(viable, key=sort_key)


def build_download_filename(niche: str, source_query: str, video_id: int, file_type: str) -> str:
    base = slugify(f"{niche} {source_query} pexels {video_id}")
    suffix = ".mp4" if file_type.lower() == "video/mp4" else ".bin"
    return f"{base}{suffix}"


def download_file(api_key: str, url: str, dest_path: Path) -> Path:
    ensure_dir(dest_path.parent)
    req = Request(url, headers={"Authorization": api_key, "User-Agent": "EchoScript/1.0"})
    with urlopen(req) as response:
        dest_path.write_bytes(response.read())
    return dest_path
