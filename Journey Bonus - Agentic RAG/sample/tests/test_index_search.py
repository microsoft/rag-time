import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import agents
from models import QueryRewriteAction


class IndexSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_uses_rewritten_query_or_original_question(self):
        question = "What does the health plan cover?"
        history = [{"role": "user", "content": "Tell me about benefits"}]

        for rewritten in ("health plan coverage", None, ""):
            with self.subTest(rewritten=rewritten):
                async def results():
                    yield {"id": "1", "content": "Covered", "sourcepage": "plan.pdf", "@search.reranker_score": 3.0}
                    yield {"id": "2", "content": "Other", "sourcepage": "other.pdf", "@search.reranker_score": 1.0}

                search = AsyncMock(return_value=results())
                state = SimpleNamespace(search_client=SimpleNamespace(search=search))
                rewrite = QueryRewriteAction(query=question, message_history=history, rewritten_query=rewritten)

                with patch.object(agents, "running_in_cl", return_value=False), patch.object(
                    agents, "rewrite_query", new=AsyncMock(return_value=rewrite)
                ) as rewrite_query:
                    result = await agents.index_search(state, question, history)

                rewrite_query.assert_awaited_once_with(state, question, history)
                kwargs = search.await_args.kwargs
                expected = rewritten or question
                self.assertEqual(kwargs["search_text"], expected)
                self.assertEqual(kwargs["vector_queries"][0].text, expected)
                self.assertEqual(result.query, question)
                self.assertEqual(result.rewritten_query, rewritten)
                self.assertEqual([document.id for document in result.documents], ["1"])


if __name__ == "__main__":
    unittest.main()
