"""ContextFactory: Factory for creating ContextVariables from various sources.

Supports file-based, dynamic, and integrated context sources.
"""

from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

from .context_variable import ContextVariable, ContextType
from .context_store import ContextStore

if TYPE_CHECKING:
    from backend.services.mind_graph import MindGraph


class ContextFactory:
    """Factory for creating ContextVariables from various sources.

    Supports:
    - File-based contexts (YAML, JSON, Markdown, etc.)
    - Dynamic contexts (function-generated)
    - Memory contexts (MindGraph integration)
    - Temporal contexts (messages, calendar)
    """

    def __init__(self, store: ContextStore):
        self.store = store

    def from_file(
        self,
        path: Path,
        id: Optional[str] = None,
        name: Optional[str] = None,
        context_type: ContextType = ContextType.DOCUMENT
    ) -> ContextVariable:
        """Create context from a file.

        Args:
            path: Path to file
            id: Custom ID (defaults to path stem)
            name: Display name (defaults to filename)
            context_type: Type classification

        Returns:
            Registered ContextVariable
        """
        path = Path(path)
        return self.store.register(
            id=id or path.stem,
            name=name or path.name,
            context_type=context_type,
            source_path=path,
            metadata={"file_type": path.suffix}
        )

    def from_directory(
        self,
        directory: Path,
        pattern: str = "*",
        context_type: ContextType = ContextType.DOCUMENT,
        id_prefix: str = ""
    ) -> list[ContextVariable]:
        """Create contexts from all files in a directory.

        Args:
            directory: Directory to scan
            pattern: Glob pattern for files
            context_type: Type for all contexts
            id_prefix: Prefix for IDs

        Returns:
            List of registered ContextVariables
        """
        contexts = []
        directory = Path(directory)

        if not directory.exists():
            return contexts

        for path in directory.glob(pattern):
            if path.is_file():
                cv = self.from_file(
                    path,
                    id=f"{id_prefix}{path.stem}" if id_prefix else None,
                    context_type=context_type
                )
                contexts.append(cv)

        return contexts

    def from_function(
        self,
        id: str,
        name: str,
        loader: Callable[[], str],
        context_type: ContextType,
        metadata: Optional[dict] = None
    ) -> ContextVariable:
        """Create context from a dynamic loader function.

        The loader is called lazily when content is accessed.
        Useful for:
        - Database queries
        - API calls
        - Computed content

        Args:
            id: Unique identifier
            name: Display name
            loader: Function that returns content string
            context_type: Type classification
            metadata: Additional metadata

        Returns:
            Registered ContextVariable
        """
        return self.store.register(
            id=id,
            name=name,
            context_type=context_type,
            source_path="<dynamic>",
            metadata=metadata or {},
            content_loader=loader
        )

    def from_yaml_collection(
        self,
        base_path: Path,
        files: list[str],
        context_type: ContextType,
        id_prefix: str = ""
    ) -> list[ContextVariable]:
        """Create contexts from a set of YAML files.

        Commonly used for foundation context:
        - profile.yaml
        - preferences.yaml
        - communication_style.yaml

        Args:
            base_path: Base directory containing the files
            files: List of filenames to load
            context_type: Type for all contexts
            id_prefix: Prefix for generated IDs

        Returns:
            List of registered ContextVariables
        """
        contexts = []
        base = Path(base_path)

        for filename in files:
            path = base / filename
            if path.exists():
                cv = self.store.register(
                    id=f"{id_prefix}{path.stem}",
                    name=path.stem.replace("_", " ").title(),
                    context_type=context_type,
                    source_path=path,
                    metadata={"format": "yaml"}
                )
                contexts.append(cv)

        return contexts

    def from_mindgraph_query(
        self,
        id: str,
        name: str,
        query: str,
        graph: "MindGraph"
    ) -> ContextVariable:
        """Create context from MindGraph semantic search.

        Args:
            id: Unique identifier
            name: Display name
            query: Semantic search query
            graph: MindGraph instance

        Returns:
            ContextVariable with search results
        """
        def loader():
            try:
                # Try to access semantic_index if available
                if hasattr(graph, 'semantic_index') and graph.semantic_index:
                    results = graph.semantic_index.search(query, limit=20)
                    lines = []
                    for r in results:
                        similarity = getattr(r, 'similarity', 0.0)
                        node = getattr(r, 'node', None)
                        if node:
                            lines.append(f"[{similarity:.2f}] {node.label}")
                            if hasattr(node, 'description') and node.description:
                                lines.append(f"   {node.description}")
                    return '\n'.join(lines) if lines else "[No results found]"
                else:
                    return f"[MindGraph semantic search not available for query: {query}]"
            except Exception as e:
                return f"[Error querying MindGraph: {e}]"

        return self.from_function(
            id=id,
            name=name,
            loader=loader,
            context_type=ContextType.MEMORY,
            metadata={"query": query, "source": "mindgraph"}
        )

    def from_text(
        self,
        id: str,
        name: str,
        content: str,
        context_type: ContextType = ContextType.DOCUMENT,
        metadata: Optional[dict] = None
    ) -> ContextVariable:
        """Create context from raw text content.

        Useful for:
        - In-memory content
        - Generated content
        - Temporary contexts

        Args:
            id: Unique identifier
            name: Display name
            content: The text content
            context_type: Type classification
            metadata: Additional metadata

        Returns:
            Registered ContextVariable
        """
        return self.from_function(
            id=id,
            name=name,
            loader=lambda: content,
            context_type=context_type,
            metadata=metadata or {"source": "text"}
        )

    def from_json_file(
        self,
        path: Path,
        id: Optional[str] = None,
        name: Optional[str] = None,
        context_type: ContextType = ContextType.DOCUMENT
    ) -> ContextVariable:
        """Create context from a JSON file.

        Args:
            path: Path to JSON file
            id: Custom ID
            name: Display name
            context_type: Type classification

        Returns:
            Registered ContextVariable
        """
        path = Path(path)
        return self.store.register(
            id=id or path.stem,
            name=name or path.name,
            context_type=context_type,
            source_path=path,
            metadata={"format": "json", "file_type": ".json"}
        )
