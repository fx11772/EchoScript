from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ScriptLine:
    label: str
    text: str


@dataclass
class ScriptDraft:
    niche: str
    lines: list[ScriptLine]

    def to_dict(self) -> dict:
        return {
            "niche": self.niche,
            "lines": [asdict(line) for line in self.lines],
        }


@dataclass
class TimedToken:
    token: str
    start_s: float
    end_s: float


@dataclass
class TimedLine:
    label: str
    text: str
    start_s: float
    end_s: float
    tokens: list[TimedToken] = field(default_factory=list)


@dataclass
class ClipCandidate:
    path: Path
    score: float
    matched_terms: list[str] = field(default_factory=list)


@dataclass
class ClipPlanItem:
    line_label: str
    line_text: str
    clip_path: Path | None
    start_s: float
    end_s: float


@dataclass
class PipelineContext:
    niche: str
    broll_dir: Path
    out_dir: Path
    scripts_dir: Path
    audio_dir: Path
    alignments_dir: Path
    plans_dir: Path
    temp_dir: Path
    renders_dir: Path
    duration_seconds: int
    voice: str | None
    seed: int | None
    keep_temp: bool
    overwrite: bool
