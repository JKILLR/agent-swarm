"""Named constants for Agent Swarm backend.

These replace magic numbers throughout the codebase for better maintainability.
"""

# Chat history
MAX_RECENT_MESSAGES = 2
MAX_CONTENT_LENGTH = 1000

# Claude CLI
CLI_TIMEOUT_HOURS = 1
CLI_TIMEOUT_SECONDS = 3600

# Web fetch
MAX_FETCH_CONTENT_LENGTH = 10000
WEB_REQUEST_TIMEOUT = 15

# Search
MAX_SEARCH_RESULTS = 10

# Executor pool
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_AGENT_TIMEOUT = 600.0
DEFAULT_MAX_TURNS = 25

# WebSocket
WS_PING_INTERVAL = 30  # seconds
WS_PING_TIMEOUT = 10  # seconds

# File operations
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50MB

# Agent spawning
AGENT_SPAWN_DETECTION_KEYWORDS = [
    "Task(subagent_type",
    "subagent_type=",
    "Task tool",
]
