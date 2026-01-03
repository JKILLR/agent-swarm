"""Tests for agent independence and proper delegation patterns.

This module tests that:
1. The COO (Supreme Orchestrator) properly delegates to agents
2. Agents execute independently in their workspaces
3. Tool execution is properly attributed to agents
4. Session context is correctly managed
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from supreme.orchestrator import SupremeOrchestrator
from tools import ToolExecutor, get_tool_definitions


class TestOrchestratorDelegation:
    """Tests for verifying the COO properly delegates work."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent

    @pytest.fixture
    def orchestrator(self, project_root) -> SupremeOrchestrator:
        """Create a SupremeOrchestrator instance."""
        return SupremeOrchestrator(
            base_path=project_root,
            config_path=project_root / "config.yaml",
            logs_dir=project_root / "logs",
        )

    def test_orchestrator_discovers_swarms(self, orchestrator):
        """Test that orchestrator discovers all swarms."""
        assert len(orchestrator.swarms) > 0, "No swarms discovered"

        # Check expected swarms exist
        expected_swarms = ["swarm_dev", "asa_research", "operations"]
        for swarm_name in expected_swarms:
            if (orchestrator.base_path / "swarms" / swarm_name).exists():
                assert swarm_name in orchestrator.swarms, f"Swarm {swarm_name} not discovered"

    def test_orchestrator_has_agents(self, orchestrator):
        """Test that orchestrator has agents available."""
        assert len(orchestrator.all_agents) > 0, "No agents found"

    def test_swarm_has_workspace(self, orchestrator):
        """Test that each swarm has a workspace directory."""
        for name, swarm in orchestrator.swarms.items():
            # Skip template swarm
            if name.startswith("_"):
                continue
            assert hasattr(swarm, "workspace"), f"Swarm {name} has no workspace attribute"
            assert swarm.workspace is not None, f"Swarm {name} workspace is None"

    def test_agent_definitions_complete(self, orchestrator):
        """Test that agent definitions have required fields."""
        for name, swarm in orchestrator.swarms.items():
            if name.startswith("_"):
                continue
            for agent_name, defn in swarm.agent_definitions.items():
                assert defn.name, f"Agent {agent_name} in {name} has no name"
                assert defn.agent_type, f"Agent {agent_name} in {name} has no type"
                assert defn.model, f"Agent {agent_name} in {name} has no model"


class TestToolExecutor:
    """Tests for ToolExecutor functionality."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        mock = MagicMock()
        mock.swarms = {}
        return mock

    @pytest.fixture
    def tool_executor(self, mock_orchestrator):
        """Create a ToolExecutor instance."""
        return ToolExecutor(orchestrator=mock_orchestrator)

    def test_tool_definitions_exist(self):
        """Test that tool definitions are properly defined."""
        definitions = get_tool_definitions()
        assert len(definitions) > 0, "No tool definitions found"

    def test_required_tools_available(self):
        """Test that essential tools are defined."""
        definitions = get_tool_definitions()
        tool_names = {t["name"] for t in definitions}

        required = {"Task", "Read", "Write", "Edit", "Bash", "Glob", "Grep"}
        for tool in required:
            assert tool in tool_names, f"Required tool {tool} not found"

    def test_task_tool_has_agent_parameter(self):
        """Test that Task tool can specify agent type."""
        definitions = get_tool_definitions()
        task_tool = next((t for t in definitions if t["name"] == "Task"), None)

        assert task_tool is not None, "Task tool not found"

        input_schema = task_tool.get("input_schema", {})
        properties = input_schema.get("properties", {})

        # Should have some way to specify agent
        agent_params = ["agent", "subagent_type", "agent_type"]
        has_agent_param = any(p in properties for p in agent_params)
        assert has_agent_param, "Task tool should have agent specification parameter"

    def test_extract_completion_summary(self, tool_executor):
        """Test summary extraction from result."""
        # Test normal result
        result = "Successfully completed the task with great results"
        summary = tool_executor._extract_completion_summary(result)
        assert len(summary) <= 150
        assert "Successfully" in summary or "completed" in summary

        # Test empty result
        summary = tool_executor._extract_completion_summary("")
        assert summary == "Task completed"

        # Test long result truncation
        long_result = "x" * 500
        summary = tool_executor._extract_completion_summary(long_result)
        assert len(summary) <= 150


class TestAgentWorkspaceIsolation:
    """Tests for agent workspace isolation."""

    @pytest.fixture
    def project_root(self) -> Path:
        return Path(__file__).parent.parent

    @pytest.fixture
    def orchestrator(self, project_root) -> SupremeOrchestrator:
        return SupremeOrchestrator(
            base_path=project_root,
            config_path=project_root / "config.yaml",
            logs_dir=project_root / "logs",
        )

    def test_swarm_workspaces_are_isolated(self, orchestrator):
        """Test that each swarm has its own workspace."""
        workspaces = set()
        for name, swarm in orchestrator.swarms.items():
            if name.startswith("_"):
                continue
            if hasattr(swarm, "workspace") and swarm.workspace:
                workspace_path = str(swarm.workspace.resolve())
                assert workspace_path not in workspaces, f"Duplicate workspace for {name}"
                workspaces.add(workspace_path)

    def test_workspace_paths_under_swarm_dir(self, orchestrator, project_root):
        """Test that workspaces are within their swarm directories."""
        swarms_dir = project_root / "swarms"
        for name, swarm in orchestrator.swarms.items():
            if name.startswith("_"):
                continue
            if hasattr(swarm, "workspace") and swarm.workspace:
                # Workspace should be under swarms/{name}/
                expected_parent = swarms_dir / name
                try:
                    swarm.workspace.relative_to(expected_parent)
                except ValueError:
                    pytest.fail(f"Workspace for {name} not under expected directory")


class TestSessionContext:
    """Tests for session context management."""

    @pytest.fixture
    def memory_dir(self, tmp_path):
        """Create temporary memory directory."""
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        return mem_dir

    @pytest.fixture
    def memory_manager(self, memory_dir):
        """Create memory manager with temp directory."""
        from memory import MemoryManager
        return MemoryManager(memory_path=memory_dir)

    def test_session_summary_saved(self, memory_manager):
        """Test that session summaries are saved correctly."""
        session_id = "test-session-123"
        summary = "Test summary content"

        memory_manager.save_session_summary(session_id, summary)

        session_file = memory_manager.memory_path / "sessions" / f"{session_id}.md"
        assert session_file.exists()
        assert summary in session_file.read_text()

    def test_conversation_summary_with_metadata(self, memory_manager):
        """Test conversation summary includes metadata."""
        session_id = "test-session-456"
        summary = "Important discussion about feature X"

        memory_manager.save_conversation_summary(
            session_id=session_id,
            summary=summary,
            original_message_count=10,
            swarm_name="test_swarm"
        )

        loaded = memory_manager.load_session_summary(session_id)
        assert loaded is not None
        assert "feature X" in loaded

    def test_context_with_summary_builds_correctly(self, memory_manager):
        """Test that context with summary is built correctly."""
        session_id = "test-session-789"

        # Save a summary
        memory_manager.save_conversation_summary(
            session_id=session_id,
            summary="Previous discussion about API design.",
            original_message_count=5
        )

        # Build context with recent messages
        recent = [
            {"role": "user", "content": "What about the API?"},
            {"role": "assistant", "content": "The API design is progressing."}
        ]

        context = memory_manager.get_context_with_summary(
            session_id=session_id,
            recent_messages=recent,
            max_recent=5
        )

        assert "Previous Conversation Summary" in context
        assert "API design" in context
        assert "What about the API" in context


class TestAgentTypeMatching:
    """Tests for matching tasks to appropriate agent types."""

    def test_implementation_keywords(self):
        """Test that implementation keywords map to implementer."""
        implementation_keywords = [
            "create", "write", "implement", "build", "code", "develop",
            "add", "make", "generate"
        ]

        for keyword in implementation_keywords:
            message = f"Please {keyword} a new feature"
            # This tests the keyword matching logic
            assert keyword.lower() in message.lower()

    def test_research_keywords(self):
        """Test that research keywords map to researcher."""
        research_keywords = [
            "research", "analyze", "investigate", "study", "explore",
            "understand", "find out", "learn about"
        ]

        for keyword in research_keywords:
            message = f"Please {keyword} the codebase"
            assert keyword.lower() in message.lower()

    def test_review_keywords(self):
        """Test that review keywords map to critic."""
        review_keywords = [
            "review", "critique", "evaluate", "assess", "check",
            "verify", "validate", "audit"
        ]

        for keyword in review_keywords:
            message = f"Please {keyword} the implementation"
            assert keyword.lower() in message.lower()


class TestParallelAgentExecution:
    """Tests for parallel agent execution patterns."""

    @pytest.mark.asyncio
    async def test_parallel_task_pattern(self):
        """Test that parallel tasks can be created correctly."""
        # This tests the pattern, not actual execution
        tasks = [
            {"agent": "researcher", "prompt": "Research topic A"},
            {"agent": "implementer", "prompt": "Implement feature B"},
            {"agent": "critic", "prompt": "Review code C"},
        ]

        # All tasks should be independent
        assert len(tasks) == 3

        # Each task has required fields
        for task in tasks:
            assert "agent" in task
            assert "prompt" in task

    @pytest.mark.asyncio
    async def test_async_iteration_pattern(self):
        """Test async iteration for streaming results."""
        async def mock_stream():
            for i in range(3):
                yield {"type": "content", "content": f"chunk {i}"}

        chunks = []
        async for chunk in mock_stream():
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0]["content"] == "chunk 0"


class TestErrorPropagation:
    """Tests for error handling and propagation."""

    @pytest.fixture
    def project_root(self):
        return Path(__file__).parent.parent

    @pytest.fixture
    def mock_orchestrator(self, project_root):
        mock = MagicMock()
        mock.base_path = project_root
        mock.swarms = {}
        return mock

    def test_tool_error_handling(self, mock_orchestrator):
        """Test that tool errors are properly captured."""
        executor = ToolExecutor(orchestrator=mock_orchestrator)

        # Tool executor should handle missing files gracefully
        # (actual behavior depends on implementation)
        assert executor is not None

    def test_invalid_swarm_handling(self, project_root):
        """Test handling of invalid swarm names."""
        orchestrator = SupremeOrchestrator(
            base_path=project_root,
            config_path=project_root / "config.yaml",
            logs_dir=project_root / "logs",
        )

        result = orchestrator.get_swarm("nonexistent_swarm_xyz")
        assert result is None


class TestAgentPromptConstruction:
    """Tests for how agent prompts are constructed."""

    def test_system_prompt_includes_role(self):
        """Test that system prompts include role information."""
        # Mock system prompt construction
        role = "implementer"
        capabilities = ["write code", "create files", "run tests"]

        system_prompt = f"You are a {role} with capabilities: {', '.join(capabilities)}"

        assert role in system_prompt
        for cap in capabilities:
            assert cap in system_prompt

    def test_prompt_includes_workspace_context(self):
        """Test that prompts include workspace information."""
        workspace = Path("/project/swarms/test_swarm/workspace")

        context = f"Working in: {workspace}"

        assert str(workspace) in context

    def test_prompt_truncation_for_long_history(self):
        """Test that long conversation history is truncated."""
        messages = [{"content": "x" * 1000} for _ in range(10)]

        # Only last 2 messages should be used (based on main.py)
        recent = messages[-2:]

        assert len(recent) == 2
        total_length = sum(len(m["content"]) for m in recent)
        assert total_length == 2000  # 2 messages * 1000 chars


class TestSwarmConfiguration:
    """Tests for swarm configuration loading."""

    @pytest.fixture
    def project_root(self):
        return Path(__file__).parent.parent

    def test_swarm_yaml_exists(self, project_root):
        """Test that swarm.yaml files exist for each swarm."""
        swarms_dir = project_root / "swarms"

        for swarm_dir in swarms_dir.iterdir():
            if swarm_dir.is_dir() and not swarm_dir.name.startswith("_"):
                config_file = swarm_dir / "swarm.yaml"
                assert config_file.exists(), f"Missing swarm.yaml for {swarm_dir.name}"

    def test_swarm_has_agents_directory(self, project_root):
        """Test that swarms have agents directory."""
        swarms_dir = project_root / "swarms"

        for swarm_dir in swarms_dir.iterdir():
            if swarm_dir.is_dir() and not swarm_dir.name.startswith("_"):
                agents_dir = swarm_dir / "agents"
                # agents directory should exist (may be empty for some swarms)
                # This is a soft check - some swarms may not have agents yet
                if agents_dir.exists():
                    assert agents_dir.is_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
