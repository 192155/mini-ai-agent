"""
test_memory.py - Unit Tests for Conversation Memory
"""

import unittest
from memory import ConversationMemory


class TestMemory(unittest.TestCase):

    def test_add_and_retrieve(self):
        """Test recording turns and retrieving history."""
        mem = ConversationMemory(max_turns=3)
        mem.add_user_message("Hello")
        mem.add_assistant_message("Hi!")

        history = mem.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "Hello")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "Hi!")

    def test_buffer_limit(self):
        """Test that memory buffer does not exceed max_turns * 2."""
        mem = ConversationMemory(max_turns=2)
        for i in range(10):
            mem.add_user_message(f"User message {i}")
            mem.add_assistant_message(f"Assistant response {i}")

        self.assertLessEqual(len(mem), 4)

    def test_clear_memory(self):
        """Test wiping conversation history."""
        mem = ConversationMemory()
        mem.add_user_message("Test")
        self.assertEqual(len(mem), 1)
        mem.clear_history()
        self.assertEqual(len(mem), 0)


if __name__ == "__main__":
    unittest.main()
