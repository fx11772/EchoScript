# Agent Guide: ai_transcribe_repo

## Primary Objective
Build and maintain a **TikTok video generation pipeline** with this exact workflow:
1. Generate script from CLI-provided niche.
2. Convert script to voice and obtain timestamps through forced alignment.
3. Select B-roll clips from local disk via filename/tag matching against script keywords.
4. Assemble 9:16 TikTok video in catchy short-form pacing.
5. Burn subtitles onto final video.

Anything outside this workflow is out of scope.

## Scope Constraints
- Use Python orchestration and FFmpeg CLI for media operations.
- Phase 1 input method is CLI args only.
- Keep existing `transcribe_folder.py` as a side tool; do not remove it.
- Do not add unrelated features while implementing the TikTok pipeline.

## Current Codebase Snapshot
- Existing side tool entrypoint: `transcribe_folder.py` (audio transcription utility).
- Tests: `tests/test_transcribe_folder.py`.
- CI workflow: `.github/workflows/ci-cd.yml`.
- Primary docs: `README.md`.

## Subtitle Rules (Non-Negotiable)
- 3-8 words per line.
- One idea per line.
- Emotional/high-impact wording.
- Fast-paced short-form tone: dramatic, motivational, slightly provocative.
- Structure:
  - `HOOK`
  - `LINE` (2-4)
  - `PAYOFF`
  - `CTA`

## Implementation Direction
- Add a new entrypoint for TikTok generation (separate from transcription script).
- Build modular components for:
  - script generation
  - TTS + forced alignment
  - keyword extraction and B-roll matching
  - FFmpeg assembly
  - subtitle rendering/burn-in
- Keep deterministic output and explicit CLI logs.

## Guardrails For Future Agents
- Prefer incremental, testable changes with explicit CLI flags.
- Add or update tests for every new behavior.
- Preserve backward compatibility of `transcribe_folder.py`.
- Update `README.md` and implementation plan docs when behavior changes.
