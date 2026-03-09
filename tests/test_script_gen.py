import unittest
from unittest.mock import MagicMock, patch

from tiktok_pipeline.models import ScriptDraft, ScriptLine
from tiktok_pipeline import script_gen as sg
from tiktok_pipeline.script_gen import (
    _build_messages,
    _variation_brief,
    generate_script,
    parse_script_response,
    validate_script_format,
)


class TestScriptGeneration(unittest.TestCase):
    def _valid_response(self) -> str:
        return "\n".join(
            [
                "HOOK: What if luxury rewires status today",
                "LINE: Wealth signals speak before words",
                "LINE: Luxury habits change your energy",
                "LINE: Quiet details reveal real power",
                "PAYOFF: Presence shapes every room instantly",
                "CTA: Comment follow and share now",
            ]
        )

    def test_parse_script_response_has_required_structure(self):
        script = parse_script_response("fitness motivation", self._valid_response())
        labels = [line.label for line in script.lines]
        self.assertEqual(labels[0], "HOOK")
        self.assertEqual(labels[-2], "PAYOFF")
        self.assertEqual(labels[-1], "CTA")
        self.assertGreaterEqual(labels.count("LINE"), 2)
        self.assertLessEqual(labels.count("LINE"), 4)

    def test_parse_script_response_word_counts_are_within_limits(self):
        script = parse_script_response("startup growth", self._valid_response())

        for line in script.lines:
            count = len(line.text.split())
            self.assertGreaterEqual(count, 3)
            self.assertLessEqual(count, 8)

    def test_generate_script_uses_openai_response(self):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=self._valid_response()))]
        )

        script = generate_script("luxury lifestyle", client=client)

        self.assertEqual(script.niche, "luxury lifestyle")
        self.assertEqual(script.lines[0].label, "HOOK")
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_variation_brief_is_deterministic_with_seed(self):
        first = _variation_brief("luxury lifestyle", seed=7)
        second = _variation_brief("luxury lifestyle", seed=7)
        third = _variation_brief("luxury lifestyle", seed=8)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_build_messages_includes_variation_brief(self):
        messages = _build_messages("luxury lifestyle", seed=11)
        user_content = messages[1]["content"]

        self.assertIn("Creative angle:", user_content)
        self.assertIn("Hook instruction:", user_content)
        self.assertIn("luxury lifestyle", user_content)

    def test_generate_script_passes_seeded_variation_to_openai(self):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=self._valid_response()))]
        )

        generate_script("luxury lifestyle", client=client, seed=23)

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 1.3)
        self.assertIn("Creative angle:", kwargs["messages"][1]["content"])

    def test_generate_script_retries_invalid_response_then_succeeds(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="bad output"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content=self._valid_response()))]),
        ]

        script = generate_script("luxury lifestyle", client=client)

        self.assertEqual(script.lines[-1].label, "CTA")
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_generate_script_requires_api_key_when_client_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                generate_script("luxury lifestyle")

    def test_generate_script_raises_after_invalid_attempts(self):
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="still invalid"))]
        )

        with self.assertRaises(ValueError):
            generate_script("luxury lifestyle", client=client, model=sg.SCRIPT_MODEL)

    def test_validate_script_format_rejects_missing_hook(self):
        bad = ScriptDraft(
            niche="test",
            lines=[
                ScriptLine(label="LINE", text="This line starts incorrectly"),
                ScriptLine(label="LINE", text="Another short build line here"),
                ScriptLine(label="PAYOFF", text="Clear payoff lands right now"),
                ScriptLine(label="CTA", text="Comment follow and share now"),
            ],
        )

        with self.assertRaises(ValueError):
            validate_script_format(bad)

    def test_validate_script_format_rejects_too_many_words(self):
        bad = ScriptDraft(
            niche="test",
            lines=[
                ScriptLine(label="HOOK", text="This sentence has way too many words for subtitles now"),
                ScriptLine(label="LINE", text="Build line stays short enough"),
                ScriptLine(label="LINE", text="Second build line also short"),
                ScriptLine(label="PAYOFF", text="Payoff lands with impact"),
                ScriptLine(label="CTA", text="Comment follow and share"),
            ],
        )

        with self.assertRaises(ValueError):
            validate_script_format(bad)


if __name__ == "__main__":
    unittest.main()
