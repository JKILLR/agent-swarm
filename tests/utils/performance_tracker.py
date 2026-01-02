"""
Performance tracking utilities for agent operations.

Provides metrics collection and benchmarking capabilities.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from contextlib import contextmanager


@dataclass
class PerformanceMetric:
    """Performance metric for an operation."""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, success: bool = True, **metadata) -> None:
        """Mark operation as complete."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.metadata.update(metadata)


@dataclass
class OperationStats:
    """Statistics for an operation type."""
    operation: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0


class PerformanceTracker:
    """Tracks performance metrics for agent operations."""

    def __init__(self, metrics_file: Optional[Path] = None):
        """
        Initialize performance tracker.

        Args:
            metrics_file: File to store metrics. If None, uses .metrics.json
        """
        self.metrics_file = metrics_file or Path.cwd() / ".metrics.json"
        self.metrics: List[PerformanceMetric] = []
        self.stats: Dict[str, OperationStats] = {}
        self._load_metrics()

    def start_operation(self, operation: str, **metadata) -> PerformanceMetric:
        """
        Start tracking an operation.

        Args:
            operation: Name of the operation
            **metadata: Additional metadata to track

        Returns:
            PerformanceMetric instance
        """
        metric = PerformanceMetric(
            operation=operation,
            start_time=time.time(),
            metadata=metadata
        )
        self.metrics.append(metric)
        return metric

    @contextmanager
    def track_operation(self, operation: str, **metadata):
        """
        Context manager for tracking an operation.

        Args:
            operation: Name of the operation
            **metadata: Additional metadata to track

        Yields:
            PerformanceMetric instance

        Example:
            with tracker.track_operation("code_generation", agent="implementer"):
                # Do work
                pass
        """
        metric = self.start_operation(operation, **metadata)
        try:
            yield metric
            metric.complete(success=True)
        except Exception as e:
            metric.complete(success=False, error=str(e))
            raise
        finally:
            self._update_stats(metric)

    def get_stats(self, operation: Optional[str] = None) -> Dict[str, OperationStats]:
        """
        Get statistics for operations.

        Args:
            operation: If provided, return stats for specific operation only

        Returns:
            Dictionary of operation statistics
        """
        if operation:
            return {operation: self.stats.get(operation, OperationStats(operation))}
        return self.stats

    def get_recent_metrics(self, count: int = 10, operation: Optional[str] = None) -> List[PerformanceMetric]:
        """
        Get most recent metrics.

        Args:
            count: Number of metrics to return
            operation: If provided, filter by operation type

        Returns:
            List of recent metrics
        """
        metrics = self.metrics
        if operation:
            metrics = [m for m in metrics if m.operation == operation]
        return metrics[-count:]

    def clear_metrics(self) -> None:
        """Clear all tracked metrics."""
        self.metrics.clear()
        self.stats.clear()
        self._save_metrics()

    def export_report(self, output_file: Optional[Path] = None) -> str:
        """
        Export performance report.

        Args:
            output_file: File to write report to

        Returns:
            Report as string
        """
        report_lines = ["# Performance Report", ""]
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append(f"Total Operations: {len(self.metrics)}")
        report_lines.append("")

        # Summary statistics
        report_lines.append("## Operation Statistics")
        report_lines.append("")
        for op_name, stats in sorted(self.stats.items()):
            report_lines.append(f"### {op_name}")
            report_lines.append(f"- Total Calls: {stats.total_calls}")
            report_lines.append(f"- Success Rate: {stats.successful_calls / stats.total_calls * 100:.1f}%")
            report_lines.append(f"- Avg Duration: {stats.avg_duration:.3f}s")
            report_lines.append(f"- Min Duration: {stats.min_duration:.3f}s")
            report_lines.append(f"- Max Duration: {stats.max_duration:.3f}s")
            report_lines.append("")

        report = "\n".join(report_lines)

        if output_file:
            output_file.write_text(report)

        return report

    def _update_stats(self, metric: PerformanceMetric) -> None:
        """Update statistics based on completed metric."""
        if metric.duration is None:
            return

        if metric.operation not in self.stats:
            self.stats[metric.operation] = OperationStats(operation=metric.operation)

        stats = self.stats[metric.operation]
        stats.total_calls += 1

        if metric.success:
            stats.successful_calls += 1
        else:
            stats.failed_calls += 1

        stats.total_duration += metric.duration
        stats.avg_duration = stats.total_duration / stats.total_calls
        stats.min_duration = min(stats.min_duration, metric.duration)
        stats.max_duration = max(stats.max_duration, metric.duration)

        self._save_metrics()

    def _load_metrics(self) -> None:
        """Load metrics from file."""
        if not self.metrics_file.exists():
            return

        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
                self.metrics = [
                    PerformanceMetric(**m)
                    for m in data.get('metrics', [])
                ]
                self.stats = {
                    k: OperationStats(**v)
                    for k, v in data.get('stats', {}).items()
                }
        except Exception:
            pass

    def _save_metrics(self) -> None:
        """Save metrics to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                data = {
                    'metrics': [asdict(m) for m in self.metrics[-1000:]],  # Keep last 1000
                    'stats': {k: asdict(v) for k, v in self.stats.items()}
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save metrics: {e}")
