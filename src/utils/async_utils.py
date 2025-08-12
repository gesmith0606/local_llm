"""Async utilities for the AI image generation project."""

import asyncio
import logging
from typing import Any, Awaitable, Callable, List, Optional, TypeVar
from concurrent.futures import ThreadPoolExecutor
import functools

logger = logging.getLogger(__name__)

T = TypeVar('T')


def run_async(coro: Awaitable[T]) -> T:
    """Run async function in sync context.
    
    Args:
        coro: Coroutine to run
        
    Returns:
        Result of the coroutine
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running, create one
        return asyncio.run(coro)
    else:
        # Event loop is running, need to use thread pool
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()


async def gather_with_progress(
    tasks: List[Awaitable[T]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    return_exceptions: bool = False
) -> List[T]:
    """Run multiple async tasks with progress tracking.
    
    Args:
        tasks: List of awaitable tasks
        progress_callback: Optional progress callback
        return_exceptions: Whether to return exceptions instead of raising
        
    Returns:
        List of results
    """
    if not tasks:
        return []
    
    total = len(tasks)
    completed = 0
    results = [None] * total
    
    async def run_with_progress(index: int, task: Awaitable[T]) -> T:
        nonlocal completed
        try:
            result = await task
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
            return result
        except Exception as e:
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
            if return_exceptions:
                return e
            raise
    
    # Run all tasks concurrently
    wrapped_tasks = [run_with_progress(i, task) for i, task in enumerate(tasks)]
    
    if return_exceptions:
        results = await asyncio.gather(*wrapped_tasks, return_exceptions=True)
    else:
        results = await asyncio.gather(*wrapped_tasks)
    
    return results


class AsyncBatch:
    """Utility for processing items in async batches."""
    
    def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
        """Initialize batch processor.
        
        Args:
            batch_size: Number of items per batch
            max_concurrent: Maximum concurrent batches
        """
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(
        self,
        items: List[Any],
        processor: Callable[[Any], Awaitable[T]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[T]:
        """Process items in batches.
        
        Args:
            items: Items to process
            processor: Async function to process each item
            progress_callback: Optional progress callback
            
        Returns:
            List of results
        """
        if not items:
            return []
        
        async def process_single_batch(batch: List[Any], start_index: int) -> List[T]:
            async with self.semaphore:
                tasks = [processor(item) for item in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update progress
                if progress_callback:
                    progress_callback(start_index + len(batch), len(items))
                
                return batch_results
        
        # Split into batches
        batches = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batches.append((batch, i))
        
        # Process all batches
        batch_tasks = [process_single_batch(batch, start_idx) for batch, start_idx in batches]
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    **kwargs
) -> T:
    """Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        *args: Function arguments
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff_factor: Backoff multiplier
        exceptions: Exceptions to catch and retry
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt == max_retries:
                break
            
            wait_time = delay * (backoff_factor ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
            await asyncio.sleep(wait_time)
    
    raise last_exception


def async_lru_cache(maxsize: int = 128):
    """LRU cache decorator for async functions."""
    def decorator(func):
        cache = {}
        cache_order = []
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check cache
            if key in cache:
                # Move to end (most recently used)
                cache_order.remove(key)
                cache_order.append(key)
                return cache[key]
            
            # Compute result
            result = await func(*args, **kwargs)
            
            # Add to cache
            cache[key] = result
            cache_order.append(key)
            
            # Enforce maxsize
            if len(cache) > maxsize:
                oldest_key = cache_order.pop(0)
                del cache[oldest_key]
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: cache.clear() or cache_order.clear()
        wrapper.cache_info = lambda: {"size": len(cache), "maxsize": maxsize}
        
        return wrapper
    return decorator


async def timeout_after(seconds: float):
    """Create a timeout coroutine."""
    await asyncio.sleep(seconds)
    raise asyncio.TimeoutError(f"Operation timed out after {seconds} seconds")


async def run_with_timeout(coro: Awaitable[T], timeout: float) -> T:
    """Run coroutine with timeout.
    
    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        
    Returns:
        Coroutine result
        
    Raises:
        asyncio.TimeoutError: If timeout is exceeded
    """
    return await asyncio.wait_for(coro, timeout=timeout)


class AsyncContextVar:
    """Async-safe context variable."""
    
    def __init__(self, default: Any = None):
        self.default = default
        self._storage = {}
    
    def set(self, value: Any):
        """Set value for current task."""
        task = asyncio.current_task()
        if task:
            self._storage[id(task)] = value
    
    def get(self, default: Any = None) -> Any:
        """Get value for current task."""
        task = asyncio.current_task()
        if task and id(task) in self._storage:
            return self._storage[id(task)]
        return default if default is not None else self.default
    
    def delete(self):
        """Delete value for current task."""
        task = asyncio.current_task()
        if task and id(task) in self._storage:
            del self._storage[id(task)]


def to_thread(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """Convert sync function to async by running in thread pool.
    
    Args:
        func: Synchronous function
        
    Returns:
        Async wrapper function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, functools.partial(func, **kwargs), *args)
    
    return wrapper
