"""
Memory management utilities for IntelliRefactor.

Provides bounded memory processing, memory monitoring,
and optimization strategies for handling large projects.
"""

import gc
import sys
import psutil
import logging
import threading
import weakref
from typing import Iterator, List, Any, Optional, Callable, TypeVar, Generic, Dict
from dataclasses import dataclass
from collections import deque
from pathlib import Path
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float  # Available system memory
    swap_mb: float  # Swap usage
    timestamp: float


class MemoryMonitor:
    """
    Memory usage monitoring and alerting system.
    
    Provides real-time memory monitoring with configurable thresholds
    and automatic garbage collection triggers.
    """
    
    def __init__(self, warning_threshold_mb: float = 500.0, 
                 critical_threshold_mb: float = 1000.0,
                 monitoring_interval: float = 5.0):
        """
        Initialize memory monitor.
        
        Args:
            warning_threshold_mb: Memory usage threshold for warnings (MB)
            critical_threshold_mb: Memory usage threshold for critical alerts (MB)
            monitoring_interval: Monitoring check interval (seconds)
        """
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.monitoring_interval = monitoring_interval
        
        self._process = psutil.Process()
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stats_history: deque = deque(maxlen=100)  # Keep last 100 measurements
        self._callbacks: List[Callable[[MemoryStats], None]] = []
        
        # Weak references to track objects for cleanup
        self._tracked_objects: weakref.WeakSet = weakref.WeakSet()
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Started memory monitoring")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        self._stop_event.set()
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        logger.info("Stopped memory monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self._stop_event.wait(self.monitoring_interval):
            try:
                stats = self.get_current_stats()
                self._stats_history.append(stats)
                
                # Check thresholds
                if stats.rss_mb >= self.critical_threshold_mb:
                    logger.critical(f"Critical memory usage: {stats.rss_mb:.1f}MB")
                    self._trigger_emergency_cleanup()
                elif stats.rss_mb >= self.warning_threshold_mb:
                    logger.warning(f"High memory usage: {stats.rss_mb:.1f}MB")
                    self._trigger_gentle_cleanup()
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(stats)
                    except Exception as e:
                        logger.error(f"Error in memory monitor callback: {e}")
                        
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
    
    def get_current_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        try:
            # Process memory info
            memory_info = self._process.memory_info()
            memory_percent = self._process.memory_percent()
            
            # System memory info
            system_memory = psutil.virtual_memory()
            swap_memory = psutil.swap_memory()
            
            return MemoryStats(
                rss_mb=memory_info.rss / 1024 / 1024,
                vms_mb=memory_info.vms / 1024 / 1024,
                percent=memory_percent,
                available_mb=system_memory.available / 1024 / 1024,
                swap_mb=swap_memory.used / 1024 / 1024,
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return MemoryStats(0, 0, 0, 0, 0, time.time())
    
    def get_stats_history(self) -> List[MemoryStats]:
        """Get historical memory statistics."""
        return list(self._stats_history)
    
    def add_callback(self, callback: Callable[[MemoryStats], None]):
        """Add callback for memory usage events."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[MemoryStats], None]):
        """Remove callback for memory usage events."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def track_object(self, obj: Any):
        """Track an object for potential cleanup."""
        self._tracked_objects.add(obj)
    
    def _trigger_gentle_cleanup(self):
        """Trigger gentle memory cleanup."""
        logger.info("Triggering gentle memory cleanup")
        
        # Force garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")
        
        # Clear weak references to dead objects
        # This happens automatically, but we can log it
        alive_objects = len(self._tracked_objects)
        logger.debug(f"Tracking {alive_objects} objects")
    
    def _trigger_emergency_cleanup(self):
        """Trigger emergency memory cleanup."""
        logger.warning("Triggering emergency memory cleanup")
        
        # Aggressive garbage collection
        for generation in range(3):
            collected = gc.collect(generation)
            logger.debug(f"GC generation {generation}: freed {collected} objects")
        
        # Clear tracked objects (they should be weak references)
        initial_count = len(self._tracked_objects)
        # WeakSet automatically removes dead references
        final_count = len(self._tracked_objects)
        logger.info(f"Tracked objects: {initial_count} -> {final_count}")
        
        # Log memory stats after cleanup
        stats = self.get_current_stats()
        logger.info(f"Memory after cleanup: {stats.rss_mb:.1f}MB")


class BoundedMemoryProcessor(Generic[T]):
    """
    Bounded memory processor for handling large datasets.
    
    Processes data in chunks to maintain bounded memory usage,
    with configurable batch sizes and memory monitoring.
    """
    
    def __init__(self, max_memory_mb: float = 200.0, 
                 batch_size: int = 100,
                 enable_monitoring: bool = True):
        """
        Initialize bounded memory processor.
        
        Args:
            max_memory_mb: Maximum memory usage threshold (MB)
            batch_size: Default batch size for processing
            enable_monitoring: Enable memory monitoring during processing
        """
        self.max_memory_mb = max_memory_mb
        self.batch_size = batch_size
        self.enable_monitoring = enable_monitoring
        
        self._memory_monitor = MemoryMonitor(
            warning_threshold_mb=max_memory_mb * 0.8,
            critical_threshold_mb=max_memory_mb
        ) if enable_monitoring else None
        
        self._processed_count = 0
        self._batch_count = 0
    
    def process_in_batches(self, items: Iterator[T], 
                          processor: Callable[[List[T]], Any],
                          batch_size: Optional[int] = None) -> Iterator[Any]:
        """
        Process items in memory-bounded batches.
        
        Args:
            items: Iterator of items to process
            processor: Function to process each batch
            batch_size: Batch size (uses default if None)
            
        Yields:
            Results from processing each batch
        """
        if self._memory_monitor and self.enable_monitoring:
            self._memory_monitor.start_monitoring()
        
        try:
            effective_batch_size = batch_size or self.batch_size
            batch = []
            
            for item in items:
                batch.append(item)
                
                if len(batch) >= effective_batch_size:
                    # Process batch
                    result = self._process_batch(batch, processor)
                    yield result
                    
                    # Clear batch and check memory
                    batch.clear()
                    self._check_memory_and_cleanup()
            
            # Process remaining items
            if batch:
                result = self._process_batch(batch, processor)
                yield result
                
        finally:
            if self._memory_monitor and self.enable_monitoring:
                self._memory_monitor.stop_monitoring()
    
    def _process_batch(self, batch: List[T], processor: Callable[[List[T]], Any]) -> Any:
        """Process a single batch with memory tracking."""
        self._batch_count += 1
        batch_size = len(batch)
        
        logger.debug(f"Processing batch {self._batch_count} with {batch_size} items")
        
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        try:
            result = processor(batch)
            self._processed_count += batch_size
            
            processing_time = time.time() - start_time
            final_memory = self._get_memory_usage()
            memory_delta = final_memory - initial_memory
            
            logger.debug(f"Batch {self._batch_count} completed in {processing_time:.2f}s, "
                        f"memory delta: {memory_delta:+.1f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing batch {self._batch_count}: {e}")
            raise
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _check_memory_and_cleanup(self):
        """Check memory usage and perform cleanup if needed."""
        current_memory = self._get_memory_usage()
        
        if current_memory > self.max_memory_mb:
            logger.warning(f"Memory usage ({current_memory:.1f}MB) exceeds limit ({self.max_memory_mb:.1f}MB)")
            
            # Force garbage collection
            collected = gc.collect()
            new_memory = self._get_memory_usage()
            
            logger.info(f"Garbage collection: {collected} objects freed, "
                       f"memory: {current_memory:.1f}MB -> {new_memory:.1f}MB")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'processed_count': self._processed_count,
            'batch_count': self._batch_count,
            'avg_batch_size': self._processed_count / self._batch_count if self._batch_count > 0 else 0,
            'current_memory_mb': self._get_memory_usage(),
            'max_memory_mb': self.max_memory_mb
        }


class MemoryManager:
    """
    Central memory management system for IntelliRefactor.
    
    Provides memory monitoring, bounded processing, and optimization
    strategies for handling large-scale analysis tasks.
    """
    
    def __init__(self, max_memory_mb: float = 500.0):
        """
        Initialize memory manager.
        
        Args:
            max_memory_mb: Maximum memory usage threshold (MB)
        """
        self.max_memory_mb = max_memory_mb
        self.monitor = MemoryMonitor(
            warning_threshold_mb=max_memory_mb * 0.7,
            critical_threshold_mb=max_memory_mb * 0.9
        )
        
        # Register cleanup callback
        self.monitor.add_callback(self._memory_callback)
        
        # Cache for reusable objects
        self._object_cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def start(self):
        """Start memory management."""
        self.monitor.start_monitoring()
        logger.info(f"Memory manager started with {self.max_memory_mb:.1f}MB limit")
    
    def stop(self):
        """Stop memory management."""
        self.monitor.stop_monitoring()
        self.clear_cache()
        logger.info("Memory manager stopped")
    
    def _memory_callback(self, stats: MemoryStats):
        """Handle memory usage events."""
        if stats.rss_mb > self.max_memory_mb * 0.8:
            # Clear cache when memory usage is high
            cache_size = len(self._object_cache)
            if cache_size > 0:
                self.clear_cache()
                logger.info(f"Cleared cache ({cache_size} items) due to high memory usage")
    
    def create_bounded_processor(self, max_memory_mb: Optional[float] = None,
                                batch_size: int = 100) -> BoundedMemoryProcessor:
        """
        Create a bounded memory processor.
        
        Args:
            max_memory_mb: Maximum memory for processor (uses manager limit if None)
            batch_size: Batch size for processing
            
        Returns:
            Configured BoundedMemoryProcessor
        """
        effective_memory_limit = max_memory_mb or (self.max_memory_mb * 0.5)
        return BoundedMemoryProcessor(
            max_memory_mb=effective_memory_limit,
            batch_size=batch_size,
            enable_monitoring=False  # Use manager's monitoring
        )
    
    def cache_object(self, key: str, obj: Any, max_size: int = 100):
        """
        Cache an object for reuse.
        
        Args:
            key: Cache key
            obj: Object to cache
            max_size: Maximum cache size
        """
        if len(self._object_cache) >= max_size:
            # Remove oldest item (simple FIFO)
            oldest_key = next(iter(self._object_cache))
            del self._object_cache[oldest_key]
        
        self._object_cache[key] = obj
    
    def get_cached_object(self, key: str) -> Optional[Any]:
        """
        Get cached object.
        
        Args:
            key: Cache key
            
        Returns:
            Cached object or None if not found
        """
        if key in self._object_cache:
            self._cache_hits += 1
            return self._object_cache[key]
        else:
            self._cache_misses += 1
            return None
    
    def clear_cache(self):
        """Clear object cache."""
        self._object_cache.clear()
        gc.collect()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self._object_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate
        }
    
    def optimize_for_large_project(self, project_size_estimate: int) -> Dict[str, Any]:
        """
        Optimize memory settings for large project analysis.
        
        Args:
            project_size_estimate: Estimated number of files in project
            
        Returns:
            Optimization settings applied
        """
        settings = {}
        
        # Adjust batch sizes based on project size
        if project_size_estimate > 1000:
            # Large project - smaller batches
            settings['batch_size'] = 50
            settings['max_memory_per_batch'] = self.max_memory_mb * 0.2
        elif project_size_estimate > 100:
            # Medium project - moderate batches
            settings['batch_size'] = 100
            settings['max_memory_per_batch'] = self.max_memory_mb * 0.3
        else:
            # Small project - larger batches
            settings['batch_size'] = 200
            settings['max_memory_per_batch'] = self.max_memory_mb * 0.5
        
        # Adjust garbage collection frequency
        if project_size_estimate > 500:
            # More frequent GC for large projects
            gc.set_threshold(700, 10, 10)  # More aggressive
            settings['gc_threshold'] = 'aggressive'
        else:
            # Default GC settings
            gc.set_threshold(700, 10, 10)
            settings['gc_threshold'] = 'default'
        
        logger.info(f"Optimized memory settings for project size {project_size_estimate}: {settings}")
        return settings
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        current_stats = self.monitor.get_current_stats()
        history = self.monitor.get_stats_history()
        cache_stats = self.get_cache_stats()
        
        # Calculate trends
        if len(history) >= 2:
            recent_avg = sum(s.rss_mb for s in history[-5:]) / min(5, len(history))
            older_avg = sum(s.rss_mb for s in history[-10:-5]) / min(5, len(history) - 5) if len(history) > 5 else recent_avg
            trend = "increasing" if recent_avg > older_avg * 1.1 else "decreasing" if recent_avg < older_avg * 0.9 else "stable"
        else:
            trend = "unknown"
        
        return {
            'current': {
                'rss_mb': current_stats.rss_mb,
                'percent': current_stats.percent,
                'available_mb': current_stats.available_mb
            },
            'limits': {
                'max_memory_mb': self.max_memory_mb,
                'warning_threshold_mb': self.monitor.warning_threshold_mb,
                'critical_threshold_mb': self.monitor.critical_threshold_mb
            },
            'trend': trend,
            'cache': cache_stats,
            'history_points': len(history),
            'recommendations': self._generate_memory_recommendations(current_stats, trend)
        }
    
    def _generate_memory_recommendations(self, stats: MemoryStats, trend: str) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        if stats.rss_mb > self.max_memory_mb * 0.8:
            recommendations.append("Memory usage is high - consider reducing batch sizes")
        
        if trend == "increasing":
            recommendations.append("Memory usage is trending upward - monitor for potential leaks")
        
        if stats.available_mb < 500:
            recommendations.append("System memory is low - consider closing other applications")
        
        cache_stats = self.get_cache_stats()
        if cache_stats['hit_rate'] < 0.5 and cache_stats['cache_size'] > 0:
            recommendations.append("Cache hit rate is low - review caching strategy")
        
        return recommendations
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()