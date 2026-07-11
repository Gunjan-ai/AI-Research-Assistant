import unittest

from app import chunk_text, rank_chunks


class PDFQATestCase(unittest.TestCase):
    def test_chunk_text_splits_into_reasonable_segments(self):
        text = "Alpha beta gamma.\n\nDelta epsilon zeta.\n\nEta theta iota."
        chunks = chunk_text(text, chunk_size=20)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(chunk) > 0 for chunk in chunks))

    def test_rank_chunks_prioritizes_relevant_content(self):
        chunks = [
            "The meeting is scheduled for Friday.",
            "The capital of France is Paris.",
            "This document discusses machine learning.",
        ]
        ranked = rank_chunks("Where is the capital of France?", chunks)
        self.assertEqual(ranked[0], chunks[1])


if __name__ == "__main__":
    unittest.main()
