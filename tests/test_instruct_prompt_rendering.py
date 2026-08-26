import unittest

try:
    from benchmark.instruct_model import InstructModel
except ModuleNotFoundError as error:
    if error.name not in {"torch", "transformers"}:
        raise
    InstructModel = None


class NoChatTemplateTokenizer:
    bos_token = None
    chat_template = None


@unittest.skipIf(InstructModel is None, "InstructModel runtime dependencies are unavailable")
class InstructPromptRenderingTest(unittest.TestCase):
    def test_missing_chat_template_preserves_the_prompt(self):
        model = InstructModel({"params": {}})
        prompt = "Below is a fully rendered PlugAE prompt."
        self.assertEqual(
            model.render_prompts([prompt], NoChatTemplateTokenizer()), [prompt]
        )

    def test_system_prompt_is_prepended_without_inventing_roles(self):
        model = InstructModel({"params": {"system_prompt": "Be concise."}})
        self.assertEqual(
            model.render_prompts(["Question"], NoChatTemplateTokenizer()),
            ["Be concise.\n\nQuestion"],
        )


if __name__ == "__main__":
    unittest.main()
