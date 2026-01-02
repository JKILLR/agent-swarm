"""Tests for memory management system."""

import sys
from pathlib import Path

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from memory import MemoryManager


class TestMemoryManager:
    """Tests for MemoryManager class."""

    @pytest.fixture
    def temp_memory_dir(self, tmp_path):
        """Create a temporary memory directory structure."""
        memory_path = tmp_path / "memory"
        memory_path.mkdir()
        return memory_path

    @pytest.fixture
    def memory_manager(self, temp_memory_dir):
        """Create a MemoryManager with temporary directory."""
        return MemoryManager(memory_path=temp_memory_dir)

    def test_ensure_structure(self, memory_manager, temp_memory_dir):
        """Test that directory structure is created."""
        assert (temp_memory_dir / "core").exists()
        assert (temp_memory_dir / "swarms").exists()
        assert (temp_memory_dir / "sessions").exists()

    def test_read_nonexistent_file(self, memory_manager):
        """Test reading a file that doesn't exist returns empty string."""
        result = memory_manager._read_file(Path("/nonexistent/file.md"))
        assert result == ""

    def test_extract_summary(self, memory_manager):
        """Test summary extraction from markdown."""
        content = """# Title

## Summary
This is the summary section.
It has multiple lines.

## Other Section
This is not the summary.
"""
        summary = memory_manager._extract_summary(content)
        assert "This is the summary section" in summary
        assert "multiple lines" in summary
        assert "Other Section" not in summary

    def test_extract_recent_entries(self, memory_manager):
        """Test extracting recent log entries."""
        content = """# Log

## 2025-01-01
Entry 1

## 2025-01-02
Entry 2

## 2025-01-03
Entry 3
"""
        # Get last 2 entries
        recent = memory_manager._extract_recent_entries(content, n=2)
        assert "2025-01-02" in recent
        assert "2025-01-03" in recent
        # First entry might or might not be there depending on how header is handled

    def test_update_progress(self, memory_manager, temp_memory_dir):
        """Test updating progress file."""
        swarm_name = "test_swarm"
        memory_manager.update_progress(swarm_name, "Test update", "Active Work")

        # Check file was created
        progress_file = temp_memory_dir / "swarms" / swarm_name / "progress.md"
        assert progress_file.exists()

        # Check content
        content = progress_file.read_text()
        assert "Test update" in content
        assert "Active Work" in content

    def test_update_progress_multiple(self, memory_manager, temp_memory_dir):
        """Test multiple progress updates."""
        swarm_name = "test_swarm"
        memory_manager.update_progress(swarm_name, "First update", "Active Work")
        memory_manager.update_progress(swarm_name, "Second update", "Active Work")

        content = (temp_memory_dir / "swarms" / swarm_name / "progress.md").read_text()
        assert "First update" in content
        assert "Second update" in content

    def test_log_decision(self, memory_manager, temp_memory_dir):
        """Test logging a decision."""
        memory_manager.log_decision(
            title="Test Decision",
            context="We needed to decide something",
            decision="We chose option A",
            rationale="Option A is better because...",
            impact="This affects X and Y",
            owner="Test Owner",
        )

        decisions_file = temp_memory_dir / "core" / "decisions.md"
        assert decisions_file.exists()

        content = decisions_file.read_text()
        assert "Test Decision" in content
        assert "Option A" in content
        assert "Test Owner" in content

    def test_save_session_summary(self, memory_manager, temp_memory_dir):
        """Test saving a session summary."""
        session_id = "test-session-123"
        summary = "This session accomplished X, Y, and Z."

        memory_manager.save_session_summary(session_id, summary)

        session_file = temp_memory_dir / "sessions" / f"{session_id}.md"
        assert session_file.exists()
        assert summary in session_file.read_text()

    def test_save_session_summary_with_swarm(self, memory_manager, temp_memory_dir):
        """Test saving session summary with swarm association."""
        session_id = "test-session-456"
        summary = "This session accomplished X."
        swarm_name = "test_swarm"

        memory_manager.save_session_summary(session_id, summary, swarm_name=swarm_name)

        # Check session file
        session_file = temp_memory_dir / "sessions" / f"{session_id}.md"
        assert session_file.exists()

        # Check swarm history
        history_file = temp_memory_dir / "swarms" / swarm_name / "history.md"
        assert history_file.exists()
        assert summary in history_file.read_text()

    def test_update_swarm_context(self, memory_manager, temp_memory_dir):
        """Test updating swarm context."""
        swarm_name = "test_swarm"

        memory_manager.update_swarm_context(swarm_name, "Current Focus", "Working on feature X")

        context_file = temp_memory_dir / "swarms" / swarm_name / "context.md"
        assert context_file.exists()

        content = context_file.read_text()
        assert "Current Focus" in content
        assert "Working on feature X" in content

    def test_load_coo_context_empty(self, memory_manager):
        """Test loading COO context when no files exist."""
        context = memory_manager.load_coo_context()
        # Should return empty or minimal content without errors
        assert isinstance(context, str)

    def test_load_coo_context_with_files(self, memory_manager, temp_memory_dir):
        """Test loading COO context with populated files."""
        # Create vision file
        (temp_memory_dir / "core").mkdir(parents=True, exist_ok=True)
        (temp_memory_dir / "core" / "vision.md").write_text("# Vision\nOur vision is X.")

        # Create priorities file
        (temp_memory_dir / "core" / "priorities.md").write_text("# Priorities\n1. Priority A")

        context = memory_manager.load_coo_context()
        assert "Vision" in context or "vision" in context.lower()
        assert "Priority" in context or "priority" in context.lower()

    def test_load_swarm_orchestrator_context(self, memory_manager, temp_memory_dir):
        """Test loading orchestrator context for a swarm."""
        swarm_name = "test_swarm"
        swarm_path = temp_memory_dir / "swarms" / swarm_name
        swarm_path.mkdir(parents=True)

        # Create context file
        (swarm_path / "context.md").write_text("# Test Swarm\n## Mission\nDo great things.")
        (swarm_path / "progress.md").write_text("# Progress\n## Active Work\n- Task 1")

        context = memory_manager.load_swarm_orchestrator_context(swarm_name)
        assert "Test Swarm" in context or "test_swarm" in context
        assert "Mission" in context or "Active Work" in context

    def test_load_agent_context(self, memory_manager, temp_memory_dir):
        """Test loading context for an individual agent."""
        swarm_name = "test_swarm"
        swarm_path = temp_memory_dir / "swarms" / swarm_name
        swarm_path.mkdir(parents=True)

        (swarm_path / "context.md").write_text("# Test Swarm\n## Mission\nAgent mission.")

        context = memory_manager.load_agent_context(swarm_name, "test_agent")
        assert isinstance(context, str)
        # Agent should see swarm context
        if context:
            assert "Test Swarm" in context or "Mission" in context


class TestSessionSummarization:
    """Tests for session summarization functionality."""

    @pytest.fixture
    def memory_manager(self, tmp_path):
        return MemoryManager(memory_path=tmp_path / "memory")

    def test_estimate_tokens(self, memory_manager):
        """Test token estimation."""
        # Roughly 4 chars per token
        text = "a" * 400
        tokens = memory_manager.estimate_tokens(text)
        assert tokens == 100

    def test_needs_summarization_short(self, memory_manager):
        """Short conversation should not need summarization."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        assert not memory_manager.needs_summarization(messages, max_tokens=1000)

    def test_needs_summarization_long(self, memory_manager):
        """Long conversation should need summarization."""
        # Create messages that exceed token limit
        messages = [
            {"role": "user", "content": "x" * 10000},
            {"role": "assistant", "content": "y" * 10000},
        ]
        assert memory_manager.needs_summarization(messages, max_tokens=1000)

    def test_create_summary_prompt(self, memory_manager):
        """Test summary prompt generation."""
        messages = [
            {"role": "user", "content": "Can you help me?"},
            {"role": "assistant", "content": "Of course!"},
        ]
        prompt = memory_manager.create_summary_prompt(messages)

        assert "Please summarize" in prompt
        assert "Can you help me?" in prompt
        assert "Of course!" in prompt
        assert "SUMMARY:" in prompt

    def test_save_and_load_conversation_summary(self, memory_manager):
        """Test saving and loading session summaries."""
        session_id = "test-session-123"
        summary = "This conversation covered important topics X, Y, and Z."

        memory_manager.save_conversation_summary(
            session_id=session_id, summary=summary, original_message_count=10, swarm_name="test_swarm"
        )

        # Load it back
        loaded = memory_manager.load_session_summary(session_id)
        assert loaded is not None
        assert "important topics" in loaded

    def test_load_nonexistent_summary(self, memory_manager):
        """Test loading summary that doesn't exist."""
        loaded = memory_manager.load_session_summary("nonexistent-session")
        assert loaded is None

    def test_get_context_with_summary(self, memory_manager):
        """Test building context with summary."""
        session_id = "test-session-456"

        # Save a summary first
        memory_manager.save_conversation_summary(
            session_id=session_id, summary="Previous discussion about feature X.", original_message_count=20
        )

        # Get context with summary + recent messages
        recent = [
            {"role": "user", "content": "What about feature Y?"},
            {"role": "assistant", "content": "Feature Y is related to X."},
        ]

        context = memory_manager.get_context_with_summary(session_id=session_id, recent_messages=recent, max_recent=5)

        assert "Previous Conversation Summary" in context
        assert "feature X" in context
        assert "feature Y" in context

    def test_get_context_without_summary(self, memory_manager):
        """Test building context when no summary exists."""
        recent = [
            {"role": "user", "content": "Hello"},
        ]

        context = memory_manager.get_context_with_summary(
            session_id="no-such-session", recent_messages=recent, max_recent=5
        )

        # Should still include recent messages
        assert "Hello" in context
        # Should not have summary section
        assert "Previous Conversation Summary" not in context


class TestMemoryManagerEdgeCases:
    """Edge case tests for MemoryManager."""

    @pytest.fixture
    def memory_manager(self, tmp_path):
        return MemoryManager(memory_path=tmp_path / "memory")

    def test_update_progress_special_characters(self, memory_manager):
        """Test progress update with special characters."""
        memory_manager.update_progress("test", "Update with $pecial ch@racters & symbols!", "Active Work")
        # Should not raise

    def test_concurrent_updates(self, memory_manager):
        """Test that multiple rapid updates don't corrupt file."""
        for i in range(10):
            memory_manager.update_progress("test", f"Update {i}", "Active Work")

        content = (memory_manager.memory_path / "swarms" / "test" / "progress.md").read_text()
        # All updates should be present
        for i in range(10):
            assert f"Update {i}" in content
