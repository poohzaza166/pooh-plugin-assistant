import asyncio
import heapq
import inspect
from dataclasses import dataclass
from enum import IntEnum
from functools import wraps
from typing import (Any, Callable, Coroutine, Dict, Generic, List, Optional,
                    ParamSpec, TypeVar, Union)

T = TypeVar('T')
P = ParamSpec('P')
R = TypeVar('R')


class Priority(IntEnum):
    """Priority levels for subscribers (lower number = higher priority)."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass(order=True)
class PrioritizedSubscriber(Generic[P, R]):
    """Wrapper for subscribers with priority."""
    priority: Priority
    func: Callable[P, R] = None

    def __post_init__(self):
        # Ensure the actual function isn't used in comparison
        object.__setattr__(self, 'func', self.func)


class MessageBus:
    def __init__(self):
        self.subscribers: Dict[str, List[PrioritizedSubscriber]] = {}
        self.data_providers: Dict[str, Callable] = {}
        self.loop_methods: List[tuple] = []
        self.event_pushers: Dict[str, Callable] = {}
        self.running_tasks: List[asyncio.Task] = []
        self._is_running = False
        self._loop = None

    def _is_async_function(self, func: Callable) -> bool:
        """Determine if a function is asynchronous."""
        return asyncio.iscoroutinefunction(func) or inspect.iscoroutinefunction(func)

    def _ensure_async(self, func: Callable[P, R]) -> Callable[P, Coroutine[Any, Any, R]]:
        """Convert a synchronous function to asynchronous if needed."""
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if self._is_async_function(func):
                return await func(*args, **kwargs)
            else:
                # Run sync function in an executor to avoid blocking
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
        return async_wrapper

    def subscribe(self, event_name: str, priority: Priority = Priority.NORMAL):
        """Decorator to subscribe a function to an event with priority."""
        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            if event_name not in self.subscribers:
                self.subscribers[event_name] = []

            prioritized = PrioritizedSubscriber(priority=priority, func=func)
            self.subscribers[event_name].append(prioritized)

            # Sort by priority (lower number = higher priority)
            self.subscribers[event_name].sort(key=lambda x: x.priority)

            return func
        return decorator

    def unsubscribe(self, event_name: str, func: Callable) -> bool:
        """Unsubscribe a function from an event."""
        if event_name in self.subscribers:
            original_length = len(self.subscribers[event_name])
            self.subscribers[event_name] = [
                sub for sub in self.subscribers[event_name] if sub.func != func
            ]
            return len(self.subscribers[event_name]) < original_length
        return False

    async def publish_async(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """Publish an event to all subscribers asynchronously in priority order."""
        if event_name in self.subscribers:
            tasks = []
            for prioritized_sub in self.subscribers[event_name]:
                async_subscriber = self._ensure_async(prioritized_sub.func)
                task = asyncio.create_task(async_subscriber(*args, **kwargs))
                tasks.append(task)

            # Wait for all async subscribers to complete if any
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def publish(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """Synchronous version of publish for backward compatibility."""
        if event_name in self.subscribers:
            for prioritized_sub in self.subscribers[event_name]:
                subscriber = prioritized_sub.func
                if self._is_async_function(subscriber):
                    # Create a new event loop if we're not in one
                    if self._loop is None:
                        try:
                            self._loop = asyncio.get_event_loop()
                        except RuntimeError:
                            self._loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(self._loop)

                    # For async subscribers in sync context, we run them but don't wait
                    asyncio.create_task(subscriber(*args, **kwargs))
                else:
                    # Call sync subscribers directly
                    subscriber(*args, **kwargs)

    def provide_data(self, data_name: str):
        """Decorator to register a data provider function."""
        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            self.data_providers[data_name] = func
            return func
        return decorator

    async def get_data_async(self, data_name: str, *args: Any, **kwargs: Any) -> Any:
        """Get data from a registered provider asynchronously."""
        if data_name in self.data_providers:
            provider = self.data_providers[data_name]
            async_provider = self._ensure_async(provider)
            return await async_provider(*args, **kwargs)
        else:
            raise KeyError(f"No data provider found for '{data_name}'")

    def get_data(self, data_name: str, *args: Any, **kwargs: Any) -> Any:
        """Synchronous version of get_data for backward compatibility."""
        if data_name in self.data_providers:
            provider = self.data_providers[data_name]
            if self._is_async_function(provider):
                # Run async function synchronously
                if self._loop is None:
                    try:
                        self._loop = asyncio.get_event_loop()
                    except RuntimeError:
                        self._loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(self._loop)

                return self._loop.run_until_complete(provider(*args, **kwargs))
            else:
                return provider(*args, **kwargs)
        else:
            raise KeyError(f"No data provider found for '{data_name}'")

    def loop_method(self, delay: float = 0):
        """Decorator to register a method to be run in a loop."""
        def decorator(func: Callable[[], Any]) -> Callable[[], Any]:
            self.loop_methods.append((func, delay))
            return func
        return decorator

    async def _run_loop_method(self, func: Callable[[], Any], delay: float) -> None:
        """Run a loop method repeatedly with the specified delay."""
        async_func = self._ensure_async(func)
        while self._is_running:
            await async_func()

            if delay > 0:
                await asyncio.sleep(delay)
            else:
                # Yield control to allow other coroutines to run
                await asyncio.sleep(0)

    def push_event(self, event_name: str):
        """Decorator to register an event pusher function."""
        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            self.event_pushers[event_name] = func
            return func
        return decorator

    async def trigger_event_async(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """Trigger an event by calling its pusher and publishing the result asynchronously."""
        if event_name in self.event_pushers:
            pusher = self.event_pushers[event_name]
            async_pusher = self._ensure_async(pusher)
            result = await async_pusher(*args, **kwargs)

            await self.publish_async(event_name, result)
        else:
            print(f"No event pusher found for '{event_name}'")

    def trigger_event(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """Synchronous version of trigger_event for backward compatibility."""
        if event_name in self.event_pushers:
            pusher = self.event_pushers[event_name]
            if self._is_async_function(pusher):
                # Run async function synchronously
                if self._loop is None:
                    try:
                        self._loop = asyncio.get_event_loop()
                    except RuntimeError:
                        self._loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(self._loop)

                result = self._loop.run_until_complete(pusher(*args, **kwargs))
            else:
                result = pusher(*args, **kwargs)

            # Publish the result
            self.publish(event_name, result)
        else:
            print(f"No event pusher found for '{event_name}'")

    async def start(self) -> None:
        """Start running all registered loop methods."""
        self._is_running = True

        # Create tasks for all loop methods
        for func, delay in self.loop_methods:
            task = asyncio.create_task(self._run_loop_method(func, delay))
            self.running_tasks.append(task)

    async def stop(self) -> None:
        """Stop all running loop methods."""
        self._is_running = False

        # Wait for all tasks to complete
        for task in self.running_tasks:
            task.cancel()

        if self.running_tasks:
            await asyncio.gather(*self.running_tasks, return_exceptions=True)

        self.running_tasks.clear()

    def start_sync(self) -> None:
        """Synchronous version of start."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)

        self._loop.run_until_complete(self.start())

    def stop_sync(self) -> None:
        """Synchronous version of stop."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)

        self._loop.run_until_complete(self.stop())
