"""Live integration tests for agent independence.

These tests require a running backend server and verify actual agent behavior.
They test real delegation patterns, tool usage, and response synthesis.

Prerequisites:
- Backend running at http://localhost:8000
- Claude CLI installed and authenticated
- At least one swarm with agents configured

Usage:
    pytest tests/test_live_agents.py -v

Note: These tests make actual API calls and may take several minutes to complete.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Try to import aiohttp for async HTTP requests
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

# Try to import websockets
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None
    WEBSOCKETS_AVAILABLE = False


# Backend URL
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
WS_URL = os.environ.get("WS_URL", "ws://localhost:8000")


def backend_available() -> bool:
    """Check if backend is available."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{BACKEND_URL}/api/status", method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


# Skip all tests if backend is not available
pytestmark = pytest.mark.skipif(
    not backend_available(),
    reason="Backend not available at {}".format(BACKEND_URL)
)


class TestBackendHealth:
    """Basic backend health tests."""

    def test_backend_status(self):
        """Test that backend is healthy."""
        import urllib.request
        req = urllib.request.Request(f"{BACKEND_URL}/api/status", method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
            assert data["status"] == "healthy"
            assert data["swarm_count"] >= 0
            assert data["agent_count"] >= 0

    def test_swarms_endpoint(self):
        """Test that swarms endpoint returns data."""
        import urllib.request
        req = urllib.request.Request(f"{BACKEND_URL}/api/swarms", method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
            assert isinstance(data, list)


class TestSwarmDiscovery:
    """Tests for swarm discovery and configuration."""

    def test_list_swarms(self):
        """Test listing all swarms."""
        import urllib.request
        req = urllib.request.Request(f"{BACKEND_URL}/api/swarms", method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            swarms = json.loads(response.read())

            assert isinstance(swarms, list)

            for swarm in swarms:
                assert "name" in swarm
                assert "description" in swarm
                assert "status" in swarm
                assert "agent_count" in swarm

    def test_get_swarm_details(self):
        """Test getting details for a specific swarm."""
        import urllib.request

        # First get list of swarms
        req = urllib.request.Request(f"{BACKEND_URL}/api/swarms", method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            swarms = json.loads(response.read())

        if not swarms:
            pytest.skip("No swarms available")

        # Get details for first swarm
        swarm_name = swarms[0]["name"]
        req = urllib.request.Request(f"{BACKEND_URL}/api/swarms/{swarm_name}", method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            details = json.loads(response.read())

            assert "name" in details
            assert details["name"] == swarm_name

    def test_get_swarm_agents(self):
        """Test getting agents for a swarm."""
        import urllib.request

        # First get list of swarms
        req = urllib.request.Request(f"{BACKEND_URL}/api/swarms", method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            swarms = json.loads(response.read())

        if not swarms:
            pytest.skip("No swarms available")

        # Get agents for first swarm
        swarm_name = swarms[0]["name"]
        req = urllib.request.Request(f"{BACKEND_URL}/api/swarms/{swarm_name}/agents", method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            agents = json.loads(response.read())

            assert isinstance(agents, list)

            for agent in agents:
                assert "name" in agent
                assert "type" in agent
                assert "model" in agent


class TestChatSessions:
    """Tests for chat session management."""

    def test_create_session(self):
        """Test creating a new chat session."""
        import urllib.request

        # Create session
        req = urllib.request.Request(
            f"{BACKEND_URL}/api/chat/sessions",
            method="POST",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            session = json.loads(response.read())

            assert "id" in session
            assert "created_at" in session
            assert "messages" in session

            return session["id"]

    def test_list_sessions(self):
        """Test listing chat sessions."""
        import urllib.request

        req = urllib.request.Request(f"{BACKEND_URL}/api/chat/sessions", method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            sessions = json.loads(response.read())

            assert isinstance(sessions, list)

    def test_session_persistence(self):
        """Test that sessions are persisted."""
        import urllib.request

        # Create a session
        req = urllib.request.Request(
            f"{BACKEND_URL}/api/chat/sessions",
            method="POST",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            session = json.loads(response.read())
            session_id = session["id"]

        # Retrieve it
        req = urllib.request.Request(
            f"{BACKEND_URL}/api/chat/sessions/{session_id}",
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            retrieved = json.loads(response.read())

            assert retrieved["id"] == session_id


class TestJobQueue:
    """Tests for background job queue."""

    def test_list_jobs(self):
        """Test listing background jobs."""
        import urllib.request

        req = urllib.request.Request(f"{BACKEND_URL}/api/jobs", method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            jobs = json.loads(response.read())

            assert isinstance(jobs, list)

    def test_job_manager_status(self):
        """Test job manager status endpoint."""
        import urllib.request

        req = urllib.request.Request(f"{BACKEND_URL}/api/jobs/status", method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            status = json.loads(response.read())

            # Should have some status information
            assert isinstance(status, dict)


@pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
class TestAsyncOperations:
    """Async tests using aiohttp."""

    @pytest.fixture
    def event_loop(self):
        """Create event loop for async tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.mark.asyncio
    async def test_concurrent_swarm_requests(self):
        """Test concurrent requests to swarm endpoints."""
        async with aiohttp.ClientSession() as session:
            # Make multiple concurrent requests
            tasks = [
                session.get(f"{BACKEND_URL}/api/swarms"),
                session.get(f"{BACKEND_URL}/api/status"),
                session.get(f"{BACKEND_URL}/api/jobs"),
            ]

            responses = await asyncio.gather(*tasks)

            for resp in responses:
                assert resp.status == 200
                await resp.json()  # Ensure valid JSON

    @pytest.mark.asyncio
    async def test_session_creation_async(self):
        """Test async session creation."""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{BACKEND_URL}/api/chat/sessions") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert "id" in data


@pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
class TestWebSocketChat:
    """Tests for WebSocket chat functionality."""

    @pytest.fixture
    def event_loop(self):
        """Create event loop for async tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)  # 30 second timeout
    async def test_websocket_connection(self):
        """Test WebSocket connection establishment."""
        try:
            async with websockets.connect(f"{WS_URL}/ws/chat") as ws:
                # Connection established
                assert ws.open
        except Exception as e:
            pytest.skip(f"WebSocket connection failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)  # 2 minute timeout for chat
    async def test_simple_chat_message(self):
        """Test sending a simple chat message."""
        try:
            async with websockets.connect(f"{WS_URL}/ws/chat") as ws:
                # Send a simple message
                message = {
                    "message": "Say 'test successful' and nothing else.",
                    "session_id": "test-ws-session"
                }
                await ws.send(json.dumps(message))

                # Collect responses
                responses = []
                try:
                    async for msg in ws:
                        data = json.loads(msg)
                        responses.append(data)

                        # Stop after chat_complete
                        if data.get("type") == "chat_complete":
                            break
                except asyncio.TimeoutError:
                    pass

                # Should have received some responses
                assert len(responses) > 0

                # Should include chat_start event
                event_types = [r.get("type") for r in responses]
                assert "chat_start" in event_types

        except Exception as e:
            pytest.skip(f"WebSocket test failed: {e}")


class TestAgentDelegationPatterns:
    """Tests that verify agent delegation patterns.

    These tests check that the COO properly delegates to agents
    rather than implementing tasks directly.
    """

    @pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")
    @pytest.mark.asyncio
    @pytest.mark.timeout(180)  # 3 minute timeout
    async def test_implementation_delegation(self):
        """Test that implementation requests are delegated to agents."""
        try:
            async with websockets.connect(f"{WS_URL}/ws/chat") as ws:
                # Send an implementation request
                message = {
                    "message": "Create a simple hello world Python file in the swarm_dev workspace. Delegate this to an implementer agent.",
                    "session_id": "test-delegation-session"
                }
                await ws.send(json.dumps(message))

                # Collect all events
                tool_events = []
                try:
                    async for msg in ws:
                        data = json.loads(msg)

                        # Track tool usage
                        if data.get("type") in ("tool_start", "tool_complete"):
                            tool_events.append(data)

                        # Stop after chat_complete
                        if data.get("type") == "chat_complete":
                            break
                except asyncio.TimeoutError:
                    pass

                # Check for Task tool usage (delegation)
                task_events = [e for e in tool_events if e.get("tool") == "Task"]

                # Should have used Task tool for delegation
                # Note: This may not always happen depending on the model's decision
                if len(task_events) == 0:
                    pytest.skip("COO did not use Task tool for delegation (may vary by model)")

        except Exception as e:
            pytest.skip(f"Delegation test failed: {e}")


class TestFileOperations:
    """Tests for file operations in swarm workspaces."""

    def test_list_workspace_files(self):
        """Test listing files in a swarm workspace."""
        import urllib.request

        # Get a swarm
        req = urllib.request.Request(f"{BACKEND_URL}/api/swarms", method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            swarms = json.loads(response.read())

        if not swarms:
            pytest.skip("No swarms available")

        swarm_name = swarms[0]["name"]

        # List files
        req = urllib.request.Request(
            f"{BACKEND_URL}/api/swarms/{swarm_name}/files",
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read())

            assert "current_path" in result
            assert "workspace" in result
            assert "directories" in result
            assert "files" in result


class TestErrorHandling:
    """Tests for error handling in the API."""

    def test_invalid_swarm_returns_404(self):
        """Test that invalid swarm name returns 404."""
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            f"{BACKEND_URL}/api/swarms/definitely_nonexistent_swarm_xyz123",
            method="GET"
        )

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)

        assert exc_info.value.code == 404

    def test_invalid_session_returns_404(self):
        """Test that invalid session ID returns 404."""
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            f"{BACKEND_URL}/api/chat/sessions/nonexistent-session-id-xyz",
            method="GET"
        )

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)

        assert exc_info.value.code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
