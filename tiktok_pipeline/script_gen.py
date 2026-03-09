from tiktok_pipeline.models import ScriptDraft, ScriptLine


# Phase 1A scaffold: deterministic placeholder output.
def generate_script_stub(niche: str) -> ScriptDraft:
    lines = [
        ScriptLine(label="HOOK", text=f"{niche} changes everything fast"),
        ScriptLine(label="LINE", text="Most people miss this pattern"),
        ScriptLine(label="LINE", text="Use this before it is late"),
        ScriptLine(label="PAYOFF", text="Small shifts create huge momentum"),
        ScriptLine(label="CTA", text="Comment if you want part two"),
    ]
    return ScriptDraft(niche=niche, lines=lines)
