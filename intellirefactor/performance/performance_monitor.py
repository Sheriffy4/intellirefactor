"""
Performance monitoring and benchmarking utilities.

Provides comprehensive performance monitoring capabilities including
execution time, memory usage, CPU utilization, and throughput metrics.
"""

import time
import psutil
import gc
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, ContextManager
from contextlib import contextmanager
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    # Timing metrics
    execution_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

    # Memory metrics (in MB)
    initial_memory_mb: float = 0.0
    final_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    memory_delta_mb: float = 0.0

    # CPU metrics
    cpu_usage_percent: float = 0.0
    cpu_time_user: float = 0.0
    cpu_time_system: float = 0.0

    # Processing metrics
    files_processed: int = 0
    lines_processed: int = 0
    bytes_processed: int = 0
    operations_completed: int = 0

    # Throughput metrics
    throughput_files_per_sec: float = 0.0
    throughput_lines_per_sec: float = 0.0
    throughput_bytes_per_sec: float = 0.0
    throughput_ops_per_sec: float = 0.0

    # Quality metrics
    errors_encountered: int = 0
    warnings_generated: int = 0
    success_rate: float = 1.0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_throughput(self):
        """Calculate throughput metrics based on execution time."""
        if self.execution_time > 0:
            self.throughput_files_per_sec = self.files_processed / self.execution_time
            self.throughput_lines_per_sec = self.lines_processed / self.execution_time
            self.throughput_bytes_per_sec = self.bytes_processed / self.execution_time
            self.throughput_ops_per_sec = self.operations_completed / self.execution_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timing": {
                "execution_time": self.execution_time,
                "start_time": self.start_time,
                "end_time": self.end_time,
            },
            "memory": {
                "initial_mb": self.initial_memory_mb,
                "final_mb": self.final_memory_mb,
                "peak_mb": self.peak_memory_mb,
                "delta_mb": self.memory_delta_mb,
            },
            "cpu": {
                "usage_percent": self.cpu_usage_percent,
                "time_user": self.cpu_time_user,
                "time_system": self.cpu_time_system,
            },
            "processing": {
                "files_processed": self.files_processed,
                "lines_processed": self.lines_processed,
                "bytes_processed": self.bytes_processed,
                "operations_completed": self.operations_completed,
            },
            "throughput": {
                "files_per_sec": self.throughput_files_per_sec,
                "lines_per_sec": self.throughput_lines_per_sec,
                "bytes_per_sec": self.throughput_bytes_per_sec,
                "ops_per_sec": self.throughput_ops_per_sec,
            },
            "quality": {
                "errors_encountered": self.errors_encountered,
                "warnings_generated": self.warnings_generated,
                "success_rate": self.success_rate,
            },
            "metadata": self.metadata,
        }


class PerformanceMonitor:
    """
    Performance monitoring and benchmarking system.

    Provides comprehensive performance monitoring capabilities with
    context managers for easy integration and detailed metrics collection.
    """

    def __init__(self, enable_continuous_monitoring: bool = False):
        """
        Initialize performance monitor.

        Args:
            enable_continuous_monitoring: Enable continuous background monitoring
        """
        self.enable_continuous_monitoring = enable_continuous_monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._continuous_metrics: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        # Process handle for system metrics
        self._process = psutil.Process()

        if enable_continuous_monitoring:
            self.start_continuous_monitoring()

    def start_continuous_monitoring(self, interval: float = 1.0):
        """
        Start continuous background monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return

        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._continuous_monitor_worker, args=(interval,), daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Started continuous performance monitoring")

    def stop_continuous_monitoring(self):
        """Stop continuous background monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5.0)
            logger.info("Stopped continuous performance monitoring")

    def _continuous_monitor_worker(self, interval: float):
        """Worker thread for continuous monitoring."""
        while not self._stop_monitoring.wait(interval):
            try:
                snapshot = self._take_system_snapshot()
                with self._lock:
                    self._continuous_metrics.append(snapshot)
                    # Keep only last 1000 snapshots to prevent memory growth
                    if len(self._continuous_metrics) > 1000:
                        self._continuous_metrics = self._continuous_metrics[-1000:]
            except Exception as e:
                logger.warning(f"Error in continuous monitoring: {e}")

    def _take_system_snapshot(self) -> Dict[str, Any]:
        """Take a snapshot of current system metrics."""
        try:
            memory_info = self._process.memory_info()
            cpu_times = self._process.cpu_times()

            return {
                "timestamp": time.time(),
                "memory_rss_mb": memory_info.rss / 1024 / 1024,
                "memory_vms_mb": memory_info.vms / 1024 / 1024,
                "cpu_user": cpu_times.user,
                "cpu_system": cpu_times.system,
                "num_threads": self._process.num_threads(),
                "num_fds": self._process.num_fds() if hasattr(self._process, "num_fds") else 0,
            }
        except Exception as e:
            logger.warning(f"Error taking system snapshot: {e}")
            return {"timestamp": time.time(), "error": str(e)}

    def get_continuous_metrics(self) -> List[Dict[str, Any]]:
        """Get collected continuous monitoring metrics."""
        with self._lock:
            return self._continuous_metrics.copy()

    def clear_continuous_metrics(self):
        """Clear collected continuous monitoring metrics."""
        with self._lock:
            self._continuous_metrics.clear()

    @contextmanager
    def monitor_operation(
        self, operation_name: str = "operation"
    ) -> ContextManager[PerformanceMetrics]:
        """
        Context manager for monitoring a specific operation.

        Args:
            operation_name: Name of the operation being monitored

        Yields:
            PerformanceMetrics object that gets populated during execution
        """
        metrics = PerformanceMetrics()
        metrics.metadata["operation_name"] = operation_name

        # Force garbage collection before measurement
        gc.collect()

        # Get initial system state
        initial_memory = self._process.memory_info().rss / 1024 / 1024
        initial_cpu_times = self._process.cpu_times()

        metrics.initial_memory_mb = initial_memory
        metrics.start_time = time.time()

        logger.debug(f"Starting performance monitoring for: {operation_name}")

        try:
            yield metrics
        finally:
            # Get final system state
            metrics.end_time = time.time()
            metrics.execution_time = metrics.end_time - metrics.start_time

            final_memory = self._process.memory_info().rss / 1024 / 1024
            final_cpu_times = self._process.cpu_times()

            metrics.final_memory_mb = final_memory
            metrics.memory_delta_mb = final_memory - initial_memory
            metrics.peak_memory_mb = max(initial_memory, final_memory)

            # Calculate CPU usage
            cpu_time_used = (final_cpu_times.user - initial_cpu_times.user) + (
                final_cpu_times.system - initial_cpu_times.system
            )
            metrics.cpu_time_user = final_cpu_times.user - initial_cpu_times.user
            metrics.cpu_time_system = final_cpu_times.system - initial_cpu_times.system

            if metrics.execution_time > 0:
                metrics.cpu_usage_percent = (cpu_time_used / metrics.execution_time) * 100

            # Calculate throughput
            metrics.calculate_throughput()

            logger.info(f"Performance monitoring completed for: {operation_name}")
            logger.info(f"  Execution time: {metrics.execution_time:.2f}s")
            logger.info(f"  Memory delta: {metrics.memory_delta_mb:.2f}MB")
            logger.info(f"  CPU usage: {metrics.cpu_usage_percent:.1f}%")

    def benchmark_function(
        self,
        func: Callable,
        *args,
        iterations: int = 1,
        warmup_iterations: int = 0,
        **kwargs,
    ) -> List[PerformanceMetrics]:
        """
        Benchmark a function with multiple iterations.

        Args:
            func: Function to benchmark
            *args: Function arguments
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations (not measured)
            **kwargs: Function keyword arguments

        Returns:
            List of PerformanceMetrics for each iteration
        """
        results = []

        # Warmup iterations
        for i in range(warmup_iterations):
            logger.debug(f"Warmup iteration {i + 1}/{warmup_iterations}")
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Error in warmup iteration {i + 1}: {e}")

        # Benchmark iterations
        for i in range(iterations):
            with self.monitor_operation(f"benchmark_iteration_{i + 1}") as metrics:
                try:
                    result = func(*args, **kwargs)
                    metrics.metadata["iteration"] = i + 1
                    metrics.metadata["result"] = str(result)[:100]  # Truncate large results
                    metrics.operations_completed = 1
                except Exception as e:
                    metrics.errors_encountered = 1
                    metrics.success_rate = 0.0
                    metrics.metadata["error"] = str(e)
                    logger.error(f"Error in benchmark iteration {i + 1}: {e}")

                results.append(metrics)

        return results

    def analyze_benchmark_results(self, results: List[PerformanceMetrics]) -> Dict[str, Any]:
        """
        Analyze benchmark results and provide statistical summary.

        Args:
            results: List of PerformanceMetrics from benchmark runs

        Returns:
            Statistical analysis of benchmark results
        """
        if not results:
            return {}

        # Extract metrics
        execution_times = [r.execution_time for r in results]
        memory_deltas = [r.memory_delta_mb for r in results]
        cpu_usages = [r.cpu_usage_percent for r in results]
        success_rates = [r.success_rate for r in results]

        def calculate_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}

            sorted_values = sorted(values)
            n = len(values)

            return {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / n,
                "median": (
                    sorted_values[n // 2]
                    if n % 2 == 1
                    else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
                ),
                "p95": sorted_values[int(0.95 * n)] if n > 1 else sorted_values[0],
                "p99": sorted_values[int(0.99 * n)] if n > 1 else sorted_values[0],
                "std_dev": (sum((x - sum(values) / n) ** 2 for x in values) / n) ** 0.5,
            }

        return {
            "iterations": len(results),
            "execution_time": calculate_stats(execution_times),
            "memory_delta": calculate_stats(memory_deltas),
            "cpu_usage": calculate_stats(cpu_usages),
            "overall_success_rate": sum(success_rates) / len(success_rates),
            "total_errors": sum(r.errors_encountered for r in results),
            "total_warnings": sum(r.warnings_generated for r in results),
        }

    def save_metrics_to_file(self, metrics: PerformanceMetrics, filepath: Path):
        """
        Save performance metrics to a JSON file.

        Args:
            metrics: Performance metrics to save
            filepath: Path to save the metrics file
        """
        try:
            with open(filepath, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2, default=str)
            logger.info(f"Performance metrics saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving metrics to file: {e}")

    def load_metrics_from_file(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """
        Load performance metrics from a JSON file.

        Args:
            filepath: Path to load the metrics file from

        Returns:
            Loaded metrics dictionary or None if error
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metrics from file: {e}")
            return None

    def __del__(self):
        """Cleanup when monitor is destroyed."""
        if hasattr(self, "_monitoring_thread"):
            self.stop_continuous_monitoring()


# Convenience functions for quick performance monitoring
def monitor_execution_time(func: Callable) -> Callable:
    """
    Decorator to monitor execution time of a function.

    Args:
        func: Function to monitor

    Returns:
        Decorated function that logs execution time
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise

    return wrapper


def monitor_memory_usage(func: Callable) -> Callable:
    """
    Decorator to monitor memory usage of a function.

    Args:
        func: Function to monitor

    Returns:
        Decorated function that logs memory usage
    """

    def wrapper(*args, **kwargs):
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        try:
            result = func(*args, **kwargs)
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_delta = final_memory - initial_memory
            logger.info(
                f"{func.__name__} memory usage: {memory_delta:+.2f}MB (final: {final_memory:.2f}MB)"
            )
            return result
        except Exception as e:
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_delta = final_memory - initial_memory
            logger.error(f"{func.__name__} failed with memory usage: {memory_delta:+.2f}MB: {e}")
            raise

    return wrapper
