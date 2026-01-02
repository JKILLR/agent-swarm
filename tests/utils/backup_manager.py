"""
Backup management utilities for safe code modifications.

Provides automated backup and restore capabilities for files being modified.
"""

import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict


@dataclass
class BackupMetadata:
    """Metadata for a backup."""
    original_path: str
    backup_path: str
    timestamp: str
    operation: str
    checksum: Optional[str] = None


class BackupManager:
    """Manages file backups and restoration."""

    def __init__(self, backup_dir: Optional[Path] = None):
        """
        Initialize backup manager.

        Args:
            backup_dir: Directory for storing backups. If None, uses .backups/
        """
        self.backup_dir = backup_dir or Path.cwd() / ".backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.backup_dir / "metadata.json"
        self.metadata: Dict[str, BackupMetadata] = self._load_metadata()

    def create_backup(self, file_path: Path, operation: str = "modify") -> Optional[Path]:
        """
        Create a backup of a file.

        Args:
            file_path: Path to file to backup
            operation: Type of operation (modify, delete, etc.)

        Returns:
            Path to backup file, or None if backup failed
        """
        if not file_path.exists():
            return None

        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.name}.{timestamp}.backup"
        backup_path = self.backup_dir / backup_name

        try:
            # Copy file to backup location
            shutil.copy2(file_path, backup_path)

            # Store metadata
            metadata = BackupMetadata(
                original_path=str(file_path.absolute()),
                backup_path=str(backup_path.absolute()),
                timestamp=timestamp,
                operation=operation
            )
            self.metadata[str(file_path.absolute())] = metadata
            self._save_metadata()

            return backup_path

        except Exception as e:
            print(f"Backup failed for {file_path}: {e}")
            return None

    def restore_backup(self, file_path: Path) -> bool:
        """
        Restore a file from its most recent backup.

        Args:
            file_path: Path to file to restore

        Returns:
            True if restoration successful, False otherwise
        """
        file_key = str(file_path.absolute())
        if file_key not in self.metadata:
            return False

        metadata = self.metadata[file_key]
        backup_path = Path(metadata.backup_path)

        if not backup_path.exists():
            return False

        try:
            # Restore file from backup
            shutil.copy2(backup_path, file_path)
            return True

        except Exception as e:
            print(f"Restore failed for {file_path}: {e}")
            return False

    def list_backups(self, file_path: Optional[Path] = None) -> List[BackupMetadata]:
        """
        List available backups.

        Args:
            file_path: If provided, list only backups for this file

        Returns:
            List of backup metadata
        """
        if file_path:
            file_key = str(file_path.absolute())
            return [self.metadata[file_key]] if file_key in self.metadata else []

        return list(self.metadata.values())

    def cleanup_old_backups(self, days: int = 7) -> int:
        """
        Clean up backups older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of backups deleted
        """
        deleted_count = 0
        current_time = datetime.now()

        for file_key, metadata in list(self.metadata.items()):
            backup_time = datetime.strptime(metadata.timestamp, "%Y%m%d_%H%M%S")
            age = (current_time - backup_time).days

            if age > days:
                backup_path = Path(metadata.backup_path)
                if backup_path.exists():
                    backup_path.unlink()
                del self.metadata[file_key]
                deleted_count += 1

        if deleted_count > 0:
            self._save_metadata()

        return deleted_count

    def _load_metadata(self) -> Dict[str, BackupMetadata]:
        """Load backup metadata from file."""
        if not self.metadata_file.exists():
            return {}

        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return {
                    k: BackupMetadata(**v)
                    for k, v in data.items()
                }
        except Exception:
            return {}

    def _save_metadata(self) -> None:
        """Save backup metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                data = {
                    k: asdict(v)
                    for k, v in self.metadata.items()
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save backup metadata: {e}")
