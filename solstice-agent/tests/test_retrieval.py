import unittest
from unittest import mock

from helpdesk import retrieval


class ExplodingLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("BM25 search must not call the LLM")


class BM25Test(unittest.TestCase):
    def test_prefers_discriminative_query_terms(self):
        candidates = [
            "General plan information and account settings.",
            "The Pro plan costs $18 per user per month.",
        ]

        ranked = retrieval.rank_bm25("Pro plan cost per user", candidates, candidates)

        self.assertEqual(ranked[0], candidates[1])

    def test_uses_full_corpus_for_document_frequency(self):
        candidates = ["frequent frequent frequent frequent", "scarce"]
        corpus = candidates + ["frequent", "frequent", "frequent"]

        ranked = retrieval.rank_bm25("frequent scarce", candidates, corpus)

        self.assertEqual(ranked[0], "scarce")

    def test_score_ties_preserve_embedding_order(self):
        candidates = ["dense-first", "dense-second"]

        ranked = retrieval.rank_bm25("unmatched", candidates, candidates)

        self.assertEqual(ranked, candidates)

    def test_search_uses_bm25_without_calling_llm(self):
        index = [
            ("Dense first but lexical mismatch.", [3.0]),
            ("Jira imports read-only cards using one-way sync.", [2.0]),
            ("Another unrelated candidate.", [1.0]),
        ]
        llm = ExplodingLLM()

        with mock.patch("helpdesk.retrieval.build_index", return_value=index), mock.patch(
            "helpdesk.retrieval.embeddings.embed", return_value=[0.0]
        ), mock.patch(
            "helpdesk.retrieval.embeddings.similarity", side_effect=lambda query, vector: vector[0]
        ):
            result = retrieval.search(llm, "Is Jira sync one-way?", k=2)

        self.assertEqual(llm.calls, 0)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(chunk, str) for chunk in result))
        self.assertEqual(result[0], index[1][0])


if __name__ == "__main__":
    unittest.main()
