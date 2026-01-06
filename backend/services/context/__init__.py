"""Context System for Life OS.

RLM-inspired context management where content stays on disk until
explicitly requested. Agents explore via peek/grep/chunk before
deciding to load full content.

Usage:
    from backend.services.context import (
        get_context_store,
        get_context_navigator,
        ContextVariable,
        ContextType,
        ContextFactory
    )

    # Get singleton instances
    store = get_context_store()
    navigator = get_context_navigator()

    # Register a new context
    store.register(
        id="my_context",
        name="My Context",
        context_type=ContextType.DOCUMENT,
        source_path=Path("path/to/file.txt")
    )

    # Explore context
    navigator.peek("my_context")
    navigator.grep("my_context", "search_pattern")
    navigator.chunk("my_context", 0)
    navigator.load("my_context")
"""

from pathlib import Path
from typing import Optional

from .context_variable import ContextVariable, ContextType
from .context_store import ContextStore, MemoryBudgetManager
from .context_navigator import ContextNavigator
from .context_factory import ContextFactory
from .context_tools import (
    CONTEXT_TOOLS,
    STATE_TOOLS,
    BUFFER_TOOLS,
    BATCH_TOOLS,
    handle_context_tool,
    get_context_tools,
    get_state_tools,
    get_buffer_tools,
    get_batch_tools,
    get_all_context_tools,
    is_context_tool
)

__all__ = [
    # Core classes
    "ContextVariable",
    "ContextType",
    "ContextStore",
    "ContextNavigator",
    "ContextFactory",
    "MemoryBudgetManager",
    # Singleton accessors
    "get_context_store",
    "get_context_navigator",
    "get_context_factory",
    # Tool definitions
    "CONTEXT_TOOLS",
    "STATE_TOOLS",
    "BUFFER_TOOLS",
    "BATCH_TOOLS",
    # Tool accessors
    "handle_context_tool",
    "get_context_tools",
    "get_state_tools",
    "get_buffer_tools",
    "get_batch_tools",
    "get_all_context_tools",
    "is_context_tool",
]

# Singleton instances
_store: Optional[ContextStore] = None
_navigator: Optional[ContextNavigator] = None
_factory: Optional[ContextFactory] = None


def get_context_store() -> ContextStore:
    """Get the singleton context store.

    Creates the store if it doesn't exist and initializes
    default contexts from the foundation directory.

    Returns:
        The singleton ContextStore instance
    """
    global _store
    if _store is None:
        # Determine storage root relative to this file
        # backend/services/context/__init__.py -> backend -> project root
        backend_dir = Path(__file__).parent.parent.parent
        project_root = backend_dir.parent
        storage_root = project_root / "memory" / "context"

        _store = ContextStore(storage_root=storage_root)
        _initialize_default_contexts(_store)
    return _store


def get_context_navigator() -> ContextNavigator:
    """Get the singleton context navigator.

    Returns:
        The singleton ContextNavigator instance
    """
    global _navigator
    if _navigator is None:
        _navigator = ContextNavigator(store=get_context_store())
    return _navigator


def get_context_factory() -> ContextFactory:
    """Get the singleton context factory.

    Returns:
        The singleton ContextFactory instance
    """
    global _factory
    if _factory is None:
        _factory = ContextFactory(store=get_context_store())
    return _factory


def _initialize_default_contexts(store: ContextStore) -> None:
    """Register default contexts (foundation, etc.).

    Called automatically when the store is first created.
    Loads identity contexts from the foundation directory.

    Args:
        store: The ContextStore to initialize
    """
    factory = ContextFactory(store)
    foundation_path = store.storage_root / "foundation"

    if foundation_path.exists():
        factory.from_yaml_collection(
            base_path=foundation_path,
            files=["profile.yaml", "preferences.yaml", "communication_style.yaml"],
            context_type=ContextType.IDENTITY,
            id_prefix="identity:"
        )

    # Load any project contexts
    projects_path = store.storage_root / "projects"
    if projects_path.exists():
        for project_dir in projects_path.iterdir():
            if project_dir.is_dir():
                project_name = project_dir.name
                factory.from_directory(
                    directory=project_dir,
                    pattern="*.yaml",
                    context_type=ContextType.PROJECT,
                    id_prefix=f"project:{project_name}:"
                )


def reset_singletons() -> None:
    """Reset all singleton instances.

    Useful for testing or when you need to reinitialize
    with different configuration.
    """
    global _store, _navigator, _factory
    _store = None
    _navigator = None
    _factory = None
