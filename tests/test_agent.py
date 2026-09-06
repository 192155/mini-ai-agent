"""
test_agent.py - Unit Tests for AI Agent Routing & Dispatch
"""

import unittest
from agent import AIAgent
from memory import ConversationMemory
from vector_store import VectorStore
from rag import RAGPipeline
from chunker import DocumentChunk


class TestAgent(unittest.TestCase):

    def setUp(self):
        self.memory = ConversationMemory()
        self.store = VectorStore()
        # Seed test document
        self.store.add_chunks([
            DocumentChunk("Inheritance in Java allows a subclass to inherit parent class members.", "test_doc.pdf", 1, 0)
        ])
        self.rag = RAGPipeline(vector_store=self.store)
        self.agent = AIAgent(memory=self.memory, rag_pipeline=self.rag)

    def test_route_to_calculator(self):
        """Verify that math queries are dispatched to the calculator tool."""
        res = self.agent.process_message("Calculate 125 * 48")
        self.assertEqual(res["source_type"], "tool")
        self.assertIn("6000", res["answer"])

    def test_route_to_rag(self):
        """Verify that document queries are dispatched to RAG."""
        res = self.agent.process_message("What is inheritance according to my uploaded PDF?")
        self.assertEqual(res["source_type"], "rag")
        self.assertIn("Inheritance in Java", res["answer"])

    def test_memory_updated_after_turn(self):
        """Verify that agent updates memory after processing a message."""
        self.agent.process_message("Calculate 5 + 5")
        self.assertEqual(len(self.agent.memory), 2)  # 1 user + 1 assistant


if __name__ == "__main__":
    unittest.main()
