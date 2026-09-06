"""
test_pdf_rag.py - Unit Tests for PDF Extraction, Chunking, Vector Store & RAG
"""

import unittest
from chunker import chunk_text, DocumentChunk
from embeddings import EmbeddingModel
from vector_store import VectorStore
from rag import RAGPipeline


class TestPDFRAG(unittest.TestCase):

    def test_chunker(self):
        """Verify text chunking with overlap."""
        sample_text = (
            "Java is an object oriented programming language. "
            "It is platform independent and runs on JVM. "
            "Inheritance allows code reuse across parent and child classes."
        )
        chunks = chunk_text(sample_text, chunk_size=80, chunk_overlap=20, source_doc="test.pdf", page_number=1)
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(chunks[0].source_doc, "test.pdf")
        self.assertEqual(chunks[0].page_number, 1)

    def test_vector_store_search(self):
        """Verify semantic similarity ranking in vector store."""
        store = VectorStore()
        chunks = [
            DocumentChunk("Python lists are dynamic arrays.", "py.pdf", 1, 0),
            DocumentChunk("Inheritance in Java enables code reuse from base class.", "java.pdf", 2, 1),
            DocumentChunk("Binary search requires a sorted array.", "algo.pdf", 3, 2),
        ]
        store.add_chunks(chunks)

        results = store.search("Tell me about inheritance in Java", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_doc"], "java.pdf")
        self.assertIn("Inheritance", results[0]["text"])

    def test_rag_unrelated_query(self):
        """Verify that RAG returns 'not found' when topic is absent."""
        store = VectorStore()
        chunks = [DocumentChunk("Java Virtual Machine executes bytecode.", "jvm.pdf", 1, 0)]
        store.add_chunks(chunks)

        rag = RAGPipeline(vector_store=store)
        res = rag.query("How to make pizza dough at home?")
        self.assertFalse(res["found"])
        self.assertIn("could not find relevant information", res["answer"])


if __name__ == "__main__":
    unittest.main()
