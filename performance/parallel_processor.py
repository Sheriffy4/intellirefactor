"""
Parallel processing utilities for IntelliRefactor.

Provides parallel processing capabilities for CPU-intensive analysis tasks
with configurable strategies and resource management.
"""

import multiprocessing
import concurrent.futures
import threading
import logging
import time
import queue
from typing import List, Any, Callable, Optional, Iterator, Dict, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import psutil

logger = logging.getLogger(__name__)


class ProcessingStrategy(Enum):
    """Processing strategy options."""
    
    SEQUENTIAL = "sequential"
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    ADAPTIVE = "adaptive"


@dataclass
class ProcessingConfig:
    """Configuration for parallel processing."""
    
    strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE
    max_workers: Optional[int] = None
    chunk_size: int = 1
    timeout: Optional[float] = None
    memory_limit_mb: float = 500.0
    cpu_threshold: float = 80.0


@dataclass
class ProcessingResult:
    """Result from parallel processing operation."""
    
    success: bool
    results: List[Any]
    errors: List[Exception]
    processing_time: float
    worker_count: int
    strategy_used: ProcessingStrategy
    items_processed: int
    items_failed: int
    throughput_items_per_sec: float


class ParallelProcessor:
    """
    Parallel processing system for IntelliRefactor analysis tasks.
    
    Provides configurable parallel processing with automatic strategy
    selection based on task characteristics and system resources.
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize parallel processor.
        
        Args:
            config: Processing configuration (uses defaults if None)
        """
        self.config = config or ProcessingConfig()
        self._cpu_count = multiprocessing.cpu_count()
        self._available_memory_mb = psutil.virtual_memory().available / 1024 / 1024
        
        # Determine optimal worker count
        if self.config.max_workers is None:
            self.config.max_workers = min(self._cpu_count, 8)  # Cap at 8 for memory reasons
        
        logger.info(f"Parallel processor initialized: {self.config.max_workers} workers, "
                   f"strategy: {self.config.strategy.value}")
    
    def process_items(self, items: List[Any], processor_func: Callable[[Any], Any],
                     strategy: Optional[ProcessingStrategy] = None) -> ProcessingResult:
        """
        Process items using specified or configured strategy.
        
        Args:
            items: List of items to process
            processor_func: Function to process each item
            strategy: Processing strategy (uses config default if None)
            
        Returns:
            ProcessingResult with results and performance metrics
        """
        if not items:
            return ProcessingResult(
                success=True,
                results=[],
                errors=[],
                processing_time=0.0,
                worker_count=0,
                strategy_used=ProcessingStrategy.SEQUENTIAL,
                items_processed=0,
                items_failed=0,
                throughput_items_per_sec=0.0
            )
        
        effective_strategy = strategy or self.config.strategy
        
        # Auto-select strategy if adaptive
        if effective_strategy == ProcessingStrategy.ADAPTIVE:
            effective_strategy = self._select_optimal_strategy(items, processor_func)
        
        start_time = time.time()
        
        try:
            if effective_strategy == ProcessingStrategy.SEQUENTIAL:
                results, errors, worker_count = self._process_sequential(items, processor_func)
            elif effective_strategy == ProcessingStrategy.THREAD_POOL:
                results, errors, worker_count = self._process_thread_pool(items, processor_func)
            elif effective_strategy == ProcessingStrategy.PROCESS_POOL:
                results, errors, worker_count = self._process_process_pool(items, processor_func)
            else:
                raise ValueError(f"Unknown processing strategy: {effective_strategy}")
            
            processing_time = time.time() - start_time
            items_processed = len(results)
            items_failed = len(errors)
            throughput = items_processed / processing_time if processing_time > 0 else 0
            
            return ProcessingResult(
                success=items_failed == 0,
                results=results,
                errors=errors,
                processing_time=processing_time,
                worker_count=worker_count,
                strategy_used=effective_strategy,
                items_processed=items_processed,
                items_failed=items_failed,
                throughput_items_per_sec=throughput
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Processing failed with strategy {effective_strategy}: {e}")
            
            return ProcessingResult(
                success=False,
                results=[],
                errors=[e],
                processing_time=processing_time,
                worker_count=0,
                strategy_used=effective_strategy,
                items_processed=0,
                items_failed=len(items),
                throughput_items_per_sec=0.0
            )
    
    def _select_optimal_strategy(self, items: List[Any], 
                               processor_func: Callable[[Any], Any]) -> ProcessingStrategy:
        """
        Select optimal processing strategy based on task characteristics.
        
        Args:
            items: Items to be processed
            processor_func: Processing function
            
        Returns:
            Optimal processing strategy
        """
        item_count = len(items)
        
        # For small datasets, use sequential processing
        if item_count < 10:
            logger.debug("Selected sequential strategy: small dataset")
            return ProcessingStrategy.SEQUENTIAL
        
        # Check system resources
        current_cpu = psutil.cpu_percent(interval=0.1)
        current_memory = psutil.virtual_memory().percent
        
        # If system is already under load, use sequential
        if current_cpu > self.config.cpu_threshold or current_memory > 85:
            logger.debug(f"Selected sequential strategy: high system load (CPU: {current_cpu}%, Memory: {current_memory}%)")
            return ProcessingStrategy.SEQUENTIAL
        
        # Estimate memory usage per item (rough heuristic)
        estimated_memory_per_item = self._estimate_memory_per_item(items)
        total_estimated_memory = estimated_memory_per_item * self.config.max_workers
        
        # If estimated memory usage is too high, use thread pool or sequential
        if total_estimated_memory > self.config.memory_limit_mb:
            if item_count < 100:
                logger.debug("Selected sequential strategy: high memory estimate for small dataset")
                return ProcessingStrategy.SEQUENTIAL
            else:
                logger.debug("Selected thread pool strategy: high memory estimate")
                return ProcessingStrategy.THREAD_POOL
        
        # For I/O bound tasks (file processing), prefer thread pool
        if self._is_io_bound_task(processor_func):
            logger.debug("Selected thread pool strategy: I/O bound task detected")
            return ProcessingStrategy.THREAD_POOL
        
        # For CPU bound tasks with sufficient resources, use process pool
        logger.debug("Selected process pool strategy: CPU bound task with sufficient resources")
        return ProcessingStrategy.PROCESS_POOL
    
    def _estimate_memory_per_item(self, items: List[Any]) -> float:
        """
        Estimate memory usage per item (rough heuristic).
        
        Args:
            items: Items to analyze
            
        Returns:
            Estimated memory usage per item in MB
        """
        if not items:
            return 1.0  # Default estimate
        
        # Sample first item to estimate size
        sample_item = items[0]
        
        if isinstance(sample_item, (str, Path)):
            # File path - estimate based on typical file size
            if isinstance(sample_item, Path):
                try:
                    file_size_mb = sample_item.stat().st_size / 1024 / 1024
                    return max(file_size_mb * 2, 5.0)  # 2x file size + overhead, min 5MB
                except:
                    pass
            return 10.0  # Default for file processing
        
        elif hasattr(sample_item, '__len__'):
            # Estimate based on length
            try:
                length = len(sample_item)
                return max(length / 1000, 1.0)  # Rough estimate: 1MB per 1000 items
            except:
                pass
        
        return 5.0  # Conservative default
    
    def _is_io_bound_task(self, processor_func: Callable) -> bool:
        """
        Heuristic to determine if task is I/O bound.
        
        Args:
            processor_func: Processing function to analyze
            
        Returns:
            True if task appears to be I/O bound
        """
        func_name = getattr(processor_func, '__name__', str(processor_func))
        
        # Common I/O bound indicators
        io_indicators = [
            'read', 'write', 'load', 'save', 'parse', 'analyze_file',
            'process_file', 'extract', 'scan', 'index'
        ]
        
        return any(indicator in func_name.lower() for indicator in io_indicators)
    
    def _process_sequential(self, items: List[Any], 
                          processor_func: Callable[[Any], Any]) -> tuple[List[Any], List[Exception], int]:
        """Process items sequentially."""
        results = []
        errors = []
        
        for item in items:
            try:
                result = processor_func(item)
                results.append(result)
            except Exception as e:
                errors.append(e)
                logger.warning(f"Error processing item {item}: {e}")
        
        return results, errors, 1
    
    def _process_thread_pool(self, items: List[Any], 
                           processor_func: Callable[[Any], Any]) -> tuple[List[Any], List[Exception], int]:
        """Process items using thread pool."""
        results = []
        errors = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(processor_func, item): item 
                for item in items
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_item, timeout=self.config.timeout):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append(e)
                    logger.warning(f"Error processing item {item}: {e}")
        
        return results, errors, self.config.max_workers
    
    def _process_process_pool(self, items: List[Any], 
                            processor_func: Callable[[Any], Any]) -> tuple[List[Any], List[Exception], int]:
        """Process items using process pool."""
        results = []
        errors = []
        
        # Process pool requires picklable functions and data
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_item = {
                    executor.submit(processor_func, item): item 
                    for item in items
                }
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_item, timeout=self.config.timeout):
                    item = future_to_item[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        errors.append(e)
                        logger.warning(f"Error processing item {item}: {e}")
            
            return results, errors, self.config.max_workers
            
        except Exception as e:
            # Fallback to thread pool if process pool fails
            logger.warning(f"Process pool failed, falling back to thread pool: {e}")
            return self._process_thread_pool(items, processor_func)
    
    def process_in_batches(self, items: List[Any], processor_func: Callable[[List[Any]], Any],
                          batch_size: Optional[int] = None) -> ProcessingResult:
        """
        Process items in batches using parallel processing.
        
        Args:
            items: Items to process
            processor_func: Function that processes a batch of items
            batch_size: Size of each batch (uses config default if None)
            
        Returns:
            ProcessingResult with batch processing results
        """
        effective_batch_size = batch_size or self.config.chunk_size
        
        # Create batches
        batches = []
        for i in range(0, len(items), effective_batch_size):
            batch = items[i:i + effective_batch_size]
            batches.append(batch)
        
        # Process batches in parallel
        return self.process_items(batches, processor_func)
    
    def map_reduce(self, items: List[Any], map_func: Callable[[Any], Any],
                  reduce_func: Callable[[List[Any]], Any]) -> ProcessingResult:
        """
        Perform map-reduce operation on items.
        
        Args:
            items: Items to process
            map_func: Map function to apply to each item
            reduce_func: Reduce function to combine results
            
        Returns:
            ProcessingResult with final reduced result
        """
        # Map phase
        map_result = self.process_items(items, map_func)
        
        if not map_result.success:
            return map_result
        
        # Reduce phase
        start_time = time.time()
        try:
            final_result = reduce_func(map_result.results)
            reduce_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                results=[final_result],
                errors=map_result.errors,
                processing_time=map_result.processing_time + reduce_time,
                worker_count=map_result.worker_count,
                strategy_used=map_result.strategy_used,
                items_processed=map_result.items_processed,
                items_failed=map_result.items_failed,
                throughput_items_per_sec=map_result.throughput_items_per_sec
            )
            
        except Exception as e:
            reduce_time = time.time() - start_time
            logger.error(f"Reduce phase failed: {e}")
            
            return ProcessingResult(
                success=False,
                results=[],
                errors=map_result.errors + [e],
                processing_time=map_result.processing_time + reduce_time,
                worker_count=map_result.worker_count,
                strategy_used=map_result.strategy_used,
                items_processed=0,
                items_failed=len(items),
                throughput_items_per_sec=0.0
            )
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get current system information for processing decisions.
        
        Returns:
            Dictionary with system resource information
        """
        memory = psutil.virtual_memory()
        
        return {
            'cpu_count': self._cpu_count,
            'cpu_usage_percent': psutil.cpu_percent(interval=0.1),
            'memory_total_mb': memory.total / 1024 / 1024,
            'memory_available_mb': memory.available / 1024 / 1024,
            'memory_usage_percent': memory.percent,
            'max_workers_configured': self.config.max_workers,
            'strategy': self.config.strategy.value,
            'memory_limit_mb': self.config.memory_limit_mb
        }
    
    def optimize_for_project_size(self, estimated_files: int, estimated_complexity: str = "medium") -> ProcessingConfig:
        """
        Optimize processing configuration based on project characteristics.
        
        Args:
            estimated_files: Estimated number of files in project
            estimated_complexity: Complexity level ("simple", "medium", "complex")
            
        Returns:
            Optimized ProcessingConfig
        """
        # Get system resources
        cpu_count = multiprocessing.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Base configuration
        config = ProcessingConfig()
        
        # Adjust based on project size
        if estimated_files < 50:
            # Small project - use fewer workers to avoid overhead
            config.max_workers = min(2, cpu_count)
            config.chunk_size = 5
            config.strategy = ProcessingStrategy.THREAD_POOL
            config.memory_limit_mb = 100.0
            
        elif estimated_files < 200:
            # Medium project - balanced approach
            config.max_workers = min(4, cpu_count)
            config.chunk_size = 10
            config.strategy = ProcessingStrategy.ADAPTIVE
            config.memory_limit_mb = 300.0
            
        elif estimated_files < 1000:
            # Large project - more workers but careful with memory
            config.max_workers = min(6, cpu_count)
            config.chunk_size = 20
            config.strategy = ProcessingStrategy.PROCESS_POOL
            config.memory_limit_mb = 500.0
            
        else:
            # Very large project - optimize for memory efficiency
            config.max_workers = min(8, cpu_count)
            config.chunk_size = 50
            config.strategy = ProcessingStrategy.PROCESS_POOL
            config.memory_limit_mb = min(1000.0, memory_gb * 1024 * 0.3)  # 30% of system memory
        
        # Adjust based on complexity
        complexity_multipliers = {
            "simple": 1.0,
            "medium": 0.8,
            "complex": 0.6
        }
        
        multiplier = complexity_multipliers.get(estimated_complexity, 0.8)
        config.max_workers = max(1, int(config.max_workers * multiplier))
        config.memory_limit_mb *= multiplier
        
        # Adjust based on available system resources
        current_cpu = psutil.cpu_percent(interval=0.1)
        current_memory = psutil.virtual_memory().percent
        
        if current_cpu > 70:
            config.max_workers = max(1, config.max_workers // 2)
            config.strategy = ProcessingStrategy.THREAD_POOL
        
        if current_memory > 80:
            config.memory_limit_mb *= 0.5
            config.chunk_size = max(1, config.chunk_size // 2)
        
        logger.info(f"Optimized config for {estimated_files} files ({estimated_complexity}): "
                   f"{config.max_workers} workers, {config.chunk_size} chunk size, "
                   f"{config.memory_limit_mb:.0f}MB limit, {config.strategy.value} strategy")
        
        return config
    
    def process_with_progress(self, items: List[Any], processor_func: Callable[[Any], Any],
                            progress_callback: Optional[Callable[[int, int], None]] = None,
                            strategy: Optional[ProcessingStrategy] = None) -> ProcessingResult:
        """
        Process items with progress reporting.
        
        Args:
            items: Items to process
            processor_func: Processing function
            progress_callback: Callback function for progress updates (processed, total)
            strategy: Processing strategy
            
        Returns:
            ProcessingResult with progress tracking
        """
        if not items:
            return ProcessingResult(
                success=True, results=[], errors=[], processing_time=0.0,
                worker_count=0, strategy_used=ProcessingStrategy.SEQUENTIAL,
                items_processed=0, items_failed=0, throughput_items_per_sec=0.0
            )
        
        effective_strategy = strategy or self.config.strategy
        if effective_strategy == ProcessingStrategy.ADAPTIVE:
            effective_strategy = self._select_optimal_strategy(items, processor_func)
        
        start_time = time.time()
        results = []
        errors = []
        processed_count = 0
        
        try:
            if effective_strategy == ProcessingStrategy.SEQUENTIAL:
                # Sequential with progress
                for i, item in enumerate(items):
                    try:
                        result = processor_func(item)
                        results.append(result)
                        processed_count += 1
                    except Exception as e:
                        errors.append(e)
                        logger.warning(f"Error processing item {i}: {e}")
                    
                    if progress_callback:
                        progress_callback(i + 1, len(items))
                
                worker_count = 1
                
            else:
                # Parallel processing with progress tracking
                if effective_strategy == ProcessingStrategy.THREAD_POOL:
                    executor_class = concurrent.futures.ThreadPoolExecutor
                else:
                    executor_class = concurrent.futures.ProcessPoolExecutor
                
                with executor_class(max_workers=self.config.max_workers) as executor:
                    # Submit all tasks
                    future_to_item = {
                        executor.submit(processor_func, item): (i, item) 
                        for i, item in enumerate(items)
                    }
                    
                    # Collect results with progress
                    for future in concurrent.futures.as_completed(future_to_item, timeout=self.config.timeout):
                        i, item = future_to_item[future]
                        try:
                            result = future.result()
                            results.append((i, result))  # Keep order info
                            processed_count += 1
                        except Exception as e:
                            errors.append(e)
                            logger.warning(f"Error processing item {i}: {e}")
                        
                        if progress_callback:
                            progress_callback(processed_count + len(errors), len(items))
                    
                    # Sort results by original order
                    results.sort(key=lambda x: x[0])
                    results = [result for _, result in results]
                
                worker_count = self.config.max_workers
            
            processing_time = time.time() - start_time
            items_failed = len(errors)
            throughput = processed_count / processing_time if processing_time > 0 else 0
            
            return ProcessingResult(
                success=items_failed == 0,
                results=results,
                errors=errors,
                processing_time=processing_time,
                worker_count=worker_count,
                strategy_used=effective_strategy,
                items_processed=processed_count,
                items_failed=items_failed,
                throughput_items_per_sec=throughput
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Processing failed with strategy {effective_strategy}: {e}")
            
            return ProcessingResult(
                success=False,
                results=[],
                errors=[e],
                processing_time=processing_time,
                worker_count=0,
                strategy_used=effective_strategy,
                items_processed=0,
                items_failed=len(items),
                throughput_items_per_sec=0.0
            )
    
    def benchmark_strategies(self, items: List[Any], processor_func: Callable[[Any], Any],
                           strategies: Optional[List[ProcessingStrategy]] = None) -> Dict[ProcessingStrategy, ProcessingResult]:
        """
        Benchmark different processing strategies.
        
        Args:
            items: Items to process for benchmarking
            processor_func: Processing function
            strategies: Strategies to benchmark (uses all if None)
            
        Returns:
            Dictionary mapping strategies to their results
        """
        if strategies is None:
            strategies = [
                ProcessingStrategy.SEQUENTIAL,
                ProcessingStrategy.THREAD_POOL,
                ProcessingStrategy.PROCESS_POOL
            ]
        
        results = {}
        
        for strategy in strategies:
            logger.info(f"Benchmarking strategy: {strategy.value}")
            
            try:
                result = self.process_items(items, processor_func, strategy)
                results[strategy] = result
                
                logger.info(f"Strategy {strategy.value}: {result.processing_time:.2f}s, "
                           f"{result.throughput_items_per_sec:.2f} items/sec")
                
            except Exception as e:
                logger.error(f"Benchmarking failed for strategy {strategy.value}: {e}")
                results[strategy] = ProcessingResult(
                    success=False,
                    results=[],
                    errors=[e],
                    processing_time=0.0,
                    worker_count=0,
                    strategy_used=strategy,
                    items_processed=0,
                    items_failed=len(items),
                    throughput_items_per_sec=0.0
                )
        
        return results
    
    def create_adaptive_processor(self, project_path: Path) -> 'ParallelProcessor':
        """
        Create an adaptively configured processor for a specific project.
        
        Args:
            project_path: Path to project to analyze
            
        Returns:
            Optimized ParallelProcessor instance
        """
        # Estimate project characteristics
        python_files = list(project_path.rglob("*.py"))
        estimated_files = len(python_files)
        
        # Estimate complexity by sampling files
        complexity = "medium"  # default
        if python_files:
            sample_size = min(10, len(python_files))
            sample_files = python_files[:sample_size]
            
            total_lines = 0
            complex_patterns = 0
            
            for file_path in sample_files:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    lines = len(content.splitlines())
                    total_lines += lines
                    
                    # Count complexity indicators
                    if 'class ' in content:
                        complex_patterns += content.count('class ')
                    if 'def ' in content:
                        complex_patterns += content.count('def ')
                    if 'import ' in content:
                        complex_patterns += content.count('import ')
                        
                except Exception:
                    continue
            
            if sample_size > 0:
                avg_lines = total_lines / sample_size
                avg_patterns = complex_patterns / sample_size
                
                if avg_lines > 500 or avg_patterns > 20:
                    complexity = "complex"
                elif avg_lines < 100 and avg_patterns < 5:
                    complexity = "simple"
        
        # Create optimized configuration
        optimized_config = self.optimize_for_project_size(estimated_files, complexity)
        
        # Create new processor with optimized config
        return ParallelProcessor(optimized_config)


# Convenience functions for common processing patterns
def parallel_map(items: List[Any], func: Callable[[Any], Any], 
                max_workers: Optional[int] = None) -> List[Any]:
    """
    Simple parallel map function.
    
    Args:
        items: Items to process
        func: Function to apply to each item
        max_workers: Maximum number of workers
        
    Returns:
        List of results
    """
    config = ProcessingConfig(max_workers=max_workers)
    processor = ParallelProcessor(config)
    result = processor.process_items(items, func)
    
    if not result.success:
        raise RuntimeError(f"Parallel processing failed: {result.errors}")
    
    return result.results


def parallel_filter(items: List[Any], predicate: Callable[[Any], bool],
                   max_workers: Optional[int] = None) -> List[Any]:
    """
    Parallel filter function.
    
    Args:
        items: Items to filter
        predicate: Predicate function
        max_workers: Maximum number of workers
        
    Returns:
        Filtered list of items
    """
    def filter_func(item):
        return item if predicate(item) else None
    
    results = parallel_map(items, filter_func, max_workers)
    return [item for item in results if item is not None]


def parallel_reduce(items: List[Any], map_func: Callable[[Any], Any],
                   reduce_func: Callable[[List[Any]], Any],
                   max_workers: Optional[int] = None) -> Any:
    """
    Parallel map-reduce function.
    
    Args:
        items: Items to process
        map_func: Map function
        reduce_func: Reduce function
        max_workers: Maximum number of workers
        
    Returns:
        Reduced result
    """
    config = ProcessingConfig(max_workers=max_workers)
    processor = ParallelProcessor(config)
    result = processor.map_reduce(items, map_func, reduce_func)
    
    if not result.success:
        raise RuntimeError(f"Map-reduce failed: {result.errors}")
    
    return result.results[0]