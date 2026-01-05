"""Integration tests for Mind Graph conversation extraction."""

import asyncio
import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.services.conversation_memory import get_conversation_memory_service
from backend.services.mind_graph import get_mind_graph


def test_explicit_memory_extraction():
    """Test that explicit memory patterns create nodes."""
    service = get_conversation_memory_service()
    graph = get_mind_graph()

    initial_count = len(graph._nodes)

    messages = [
        {"role": "user", "content": "Remember that my favorite color is blue"},
        {"role": "assistant", "content": "I'll remember that your favorite color is blue!"},
        {"role": "user", "content": "My name is TestUser"},
    ]

    # Run the async extraction
    nodes = asyncio.run(service.process_conversation(
        session_id="test-explicit-123",
        messages=messages,
    ))

    assert len(nodes) >= 1, f"Should create at least one memory node, got {len(nodes)}"

    # Verify nodes in graph
    final_count = len(graph._nodes)
    assert final_count > initial_count, "Graph should have new nodes"

    print(f"Created {len(nodes)} nodes from conversation")
    for node in nodes:
        print(f"  - {node.node_type.value}: {node.label}")


def test_conversation_analyzer_patterns():
    """Test that ConversationAnalyzer finds explicit patterns."""
    from backend.services.conversation_analyzer import ConversationAnalyzer

    analyzer = ConversationAnalyzer()

    # Test various patterns
    test_cases = [
        ("Remember that I love pizza", "fact"),
        ("My name is John", "identity"),
        ("I prefer dark mode", "preference"),
        ("I want to learn Python", "goal"),
    ]

    for message, expected_category in test_cases:
        memories = analyzer.extract_explicit(message)
        assert len(memories) > 0, f"Should extract from: {message}"
        print(f"Pattern '{message}' -> {memories[0].category.value}")


if __name__ == "__main__":
    print("Running Mind Graph integration tests...")
    test_conversation_analyzer_patterns()
    print("\nPattern tests passed!")

    test_explicit_memory_extraction()
    print("\nExplicit extraction tests passed!")

    print("\nAll tests passed!")
