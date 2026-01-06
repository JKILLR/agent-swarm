# backend/services/context_service.py
"""Personal context system for Life OS.

Provides hierarchical context management with three layers:
1. Foundation - Core personal profile (always loaded)
2. Project - Project-specific context (lazy loaded)
3. Working - Ephemeral session context

Optimized for 8GB RAM with file-based storage and optional embeddings.
"""

import logging
import threading
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# Base paths
CONTEXT_ROOT = Path(__file__).parent.parent.parent / "memory" / "context"
FOUNDATION_PATH = CONTEXT_ROOT / "foundation"
PROJECTS_PATH = CONTEXT_ROOT / "projects"


class ContextService:
    """Manages personal context across foundation, project, and working layers.
    
    Features:
    - Foundation context always in memory (~50KB)
    - Project context lazy-loaded and cached
    - Working context for current session
    - Thread-safe operations
    """
    
    def __init__(self):
        self._foundation: Optional[dict] = None
        self._project_cache: dict[str, dict] = {}
        self._working: dict[str, Any] = {}
        self._lock = threading.Lock()
        
    def _load_yaml(self, path: Path) -> dict:
        """Load a YAML file, returning empty dict if not found."""
        if not path.exists():
            logger.warning(f"Context file not found: {path}")
            return {}
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return {}
    
    def _save_yaml(self, path: Path, data: dict) -> bool:
        """Save data to a YAML file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            logger.error(f"Error saving {path}: {e}")
            return False
    
    def _load_foundation(self) -> dict:
        """Load all foundation context files."""
        foundation = {
            "profile": self._load_yaml(FOUNDATION_PATH / "profile.yaml"),
            "communication_style": self._load_yaml(FOUNDATION_PATH / "communication_style.yaml"),
            "preferences": self._load_yaml(FOUNDATION_PATH / "preferences.yaml"),
        }
        logger.info(f"Loaded foundation context: {list(foundation.keys())}")
        return foundation
    
    def _load_project(self, project_name: str) -> dict:
        """Load project-specific context."""
        project_path = PROJECTS_PATH / project_name
        if not project_path.exists():
            logger.warning(f"Project not found: {project_name}")
            return {}
        
        project = {
            "project": self._load_yaml(project_path / "project.yaml"),
            "trades": self._load_yaml(project_path / "trades.yaml"),
            "contacts": self._load_yaml(project_path / "contacts.yaml"),
        }
        
        # Load cached doc summaries if they exist
        docs_path = project_path / "docs"
        if docs_path.exists():
            project["docs"] = {}
            for doc_file in docs_path.glob("*.json"):
                import json
                try:
                    with open(doc_file) as f:
                        project["docs"][doc_file.stem] = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading doc {doc_file}: {e}")
        
        logger.info(f"Loaded project context: {project_name}")
        return project
    
    @property
    def foundation(self) -> dict:
        """Get foundation context (lazy loaded, cached)."""
        if self._foundation is None:
            with self._lock:
                if self._foundation is None:
                    self._foundation = self._load_foundation()
        return self._foundation
    
    def get_foundation(self) -> dict:
        """Return foundation context (profile, style, preferences)."""
        return self.foundation
    
    def get_project(self, project_name: str) -> dict:
        """Get project context (lazy loaded, cached)."""
        if project_name not in self._project_cache:
            with self._lock:
                if project_name not in self._project_cache:
                    self._project_cache[project_name] = self._load_project(project_name)
        return self._project_cache[project_name]
    
    def get_working(self) -> dict:
        """Get current working context."""
        return self._working
    
    def update_working(self, key: str, value: Any) -> None:
        """Update working context with a key-value pair."""
        self._working[key] = value
    
    def set_working(self, context: dict) -> None:
        """Replace entire working context."""
        self._working = context
    
    def clear_working(self) -> None:
        """Clear working context (typically at session end)."""
        self._working = {}
    
    def get_combined_context(self, project_name: Optional[str] = None) -> dict:
        """Get merged context for LLM prompt injection.
        
        Returns a combined dict with:
        - user: Foundation context
        - project: Project context (if specified)
        - working: Current working context
        """
        context = {
            "user": self.foundation,
            "working": self._working
        }
        if project_name:
            context["project"] = self.get_project(project_name)
        return context
    
    def get_context_summary(self, project_name: Optional[str] = None) -> str:
        """Get a text summary suitable for LLM system prompts.
        
        Returns a concise summary of the user and optionally project context.
        """
        f = self.foundation
        lines = []
        
        # User info
        profile = f.get("profile", {})
        if profile:
            lines.append(f"User: {profile.get('name', 'Unknown')}")
            lines.append(f"Role: {profile.get('profession', 'Unknown')}")
            if profile.get('current_project'):
                lines.append(f"Current Project: {profile.get('current_project')}")
        
        # Communication style
        style = f.get("communication_style", {})
        if style:
            email = style.get("email", {})
            if email.get("tone"):
                lines.append(f"Communication Style: {email.get('tone')}")
        
        # Project info
        if project_name:
            project = self.get_project(project_name)
            proj = project.get("project", {})
            if proj:
                lines.append(f"\n--- Project: {proj.get('project_name', project_name)} ---")
                if proj.get("project_type"):
                    lines.append(f"Type: {proj.get('project_type')}")
                if proj.get("location", {}).get("city"):
                    lines.append(f"Location: {proj['location']['city']}")
        
        return "\n".join(lines)
    
    def update_profile(self, updates: dict) -> bool:
        """Update profile.yaml with new values."""
        profile = self._load_yaml(FOUNDATION_PATH / "profile.yaml")
        profile.update(updates)
        success = self._save_yaml(FOUNDATION_PATH / "profile.yaml", profile)
        if success:
            self._foundation = None  # Force reload
        return success
    
    def update_communication_style(self, updates: dict) -> bool:
        """Update communication_style.yaml with new values."""
        style = self._load_yaml(FOUNDATION_PATH / "communication_style.yaml")
        # Deep merge for nested keys
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(style.get(key), dict):
                style[key].update(value)
            else:
                style[key] = value
        success = self._save_yaml(FOUNDATION_PATH / "communication_style.yaml", style)
        if success:
            self._foundation = None  # Force reload
        return success
    
    def learn_phrase(self, phrase_type: str, phrase: str) -> bool:
        """Learn a new phrase (greeting, closing, transition) from user behavior."""
        style = self._load_yaml(FOUNDATION_PATH / "communication_style.yaml")
        phrases = style.get("common_phrases", {})
        
        if phrase_type not in phrases:
            phrases[phrase_type] = []
        
        # Only add if not already present
        if phrase not in phrases[phrase_type]:
            phrases[phrase_type].append(phrase)
            style["common_phrases"] = phrases
            return self._save_yaml(FOUNDATION_PATH / "communication_style.yaml", style)
        
        return True  # Already exists
    
    def list_projects(self) -> list[str]:
        """List all available projects."""
        if not PROJECTS_PATH.exists():
            return []
        return [p.name for p in PROJECTS_PATH.iterdir() if p.is_dir()]
    
    def invalidate_cache(self, project_name: Optional[str] = None) -> None:
        """Invalidate cached context to force reload.
        
        If project_name is None, invalidates all caches.
        """
        with self._lock:
            if project_name:
                self._project_cache.pop(project_name, None)
            else:
                self._foundation = None
                self._project_cache = {}


# Singleton
_context_service: Optional[ContextService] = None
_singleton_lock = threading.Lock()


def get_context_service() -> ContextService:
    """Get the singleton ContextService instance."""
    global _context_service
    if _context_service is None:
        with _singleton_lock:
            if _context_service is None:
                logger.info("Creating ContextService singleton")
                _context_service = ContextService()
    return _context_service
