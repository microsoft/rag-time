import json
import unittest
from pathlib import Path

import prompty


class PromptMessageTests(unittest.TestCase):
    def test_sample_prompts_produce_openai_chat_messages(self):
        prompts = Path(__file__).resolve().parent.parent / "prompts"
        cases = [
            ("chat_query_rewrite.prompty", {"user_query": "What benefits are covered?", "past_messages": []}, "What benefits are covered?"),
            ("chat_answer_question.prompty", {"user_query": "What benefits are covered?", "context": "Sources: dental care", "previous_answer": None, "previous_answer_reflection": None, "previous_answer_evaluations": None}, "dental care"),
            ("chat_reflect_answer.prompty", {"previous_query": "benefits", "previous_answer": "Dental care", "previous_sources": "Sources: dental care", "previous_answer_evaluation": "Relevant"}, "Dental care"),
        ]

        for filename, inputs, expected in cases:
            with self.subTest(prompt=filename):
                messages = prompty.prepare(prompty.load(str(prompts / filename)), inputs)
                json.dumps(messages)
                self.assertTrue(messages)
                self.assertTrue(all(isinstance(message, dict) and {"role", "content"} <= message.keys() for message in messages))
                self.assertIn(expected, "\n".join(message["content"] for message in messages))


if __name__ == "__main__":
    unittest.main()
