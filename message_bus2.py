import asyncio
import inspect
from functools import wraps
from typing import (Any, Callable, Coroutine, Dict, List, Optional, TypeVar,
                    Union, cast)

T = TypeVar('T')

class MessageBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.data_providers: Dict[str, Callable] = {}
        self.loop_methods: List[tuple] = []
        self.event_pushers: Dict[str, Callable] = {}
        self.running_tasks: List[asyncio.Task] = []
        self._is_running = False
        self._loop = None

    def _is_async_function(self, func: Callable) -> bool:
        """Determine if a function is asynchronous."""
        return asyncio.iscoroutinefunction(func) or inspect.iscoroutinefunction(func)

    def _ensure_async(self, func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
        """Convert a synchronous function to asynchronous if needed."""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if self._is_async_function(func):
                return await func(*args, **kwargs)
            else:
                # Run sync function in an executor to avoid blocking
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
        return async_wrapper

    def subscribe(self, event_name: str):
        """Decorator to subscribe a function to an event."""
        def decorator(func: Callable):
            if event_name not in self.subscribers:
                self.subscribers[event_name] = []
            self.subscribers[event_name].append(func)
            return func
        return decorator

    async def publish_async(self, event_name: str, *args, **kwargs):
        """Publish an event to all subscribers asynchronously."""
        if event_name in self.subscribers:
            tasks = []
            for subscriber in self.subscribers[event_name]:
                async_subscriber = self._ensure_async(subscriber)
                task = asyncio.create_task(async_subscriber(*args, **kwargs))
                tasks.append(task)
            
            # Wait for all async subscribers to complete if any
            if tasks:
                await asyncio.gather(*tasks)

    def publish(self, event_name: str, *args, **kwargs):
        """Synchronous version of publish for backward compatibility."""
        if event_name in self.subscribers:
            for subscriber in self.subscribers[event_name]:
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
        def decorator(func: Callable):
            self.data_providers[data_name] = func
            return func
        return decorator

    async def get_data_async(self, data_name: str, *args, **kwargs) -> Any:
        """Get data from a registered provider asynchronously."""
        if data_name in self.data_providers:
            provider = self.data_providers[data_name]
            async_provider = self._ensure_async(provider)
            return await async_provider(*args, **kwargs)
        else:
            raise KeyError(f"No data provider found for '{data_name}'")

    def get_data(self, data_name: str, *args, **kwargs) -> Any:
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
        def decorator(func: Callable):
            self.loop_methods.append((func, delay))
            return func
        return decorator

    async def _run_loop_method(self, func: Callable, delay: float):
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
        def decorator(func: Callable):
            self.event_pushers[event_name] = func
            return func
        return decorator

    async def trigger_event_async(self, event_name: str, *args, **kwargs):
        """Trigger an event by calling its pusher and publishing the result asynchronously."""
        if event_name in self.event_pushers:
            pusher = self.event_pushers[event_name]
            async_pusher = self._ensure_async(pusher)
            result = await async_pusher(*args, **kwargs)
            
            await self.publish_async(event_name, result)
        else:
            print(f"No event pusher found for '{event_name}'")

    def trigger_event(self, event_name: str, *args, **kwargs):
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

    async def start(self):
        """Start running all registered loop methods."""
        self._is_running = True
        
        # Create tasks for all loop methods
        for func, delay in self.loop_methods:
            task = asyncio.create_task(self._run_loop_method(func, delay))
            self.running_tasks.append(task)
    
    async def stop(self):
        """Stop all running loop methods."""
        self._is_running = False
        
        # Wait for all tasks to complete
        for task in self.running_tasks:
            task.cancel()
        
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks, return_exceptions=True)
        
        self.running_tasks.clear()

    def start_sync(self):
        """Synchronous version of start."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        
        self._loop.run_until_complete(self.start())

    def stop_sync(self):
        """Synchronous version of stop."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        
        self._loop.run_until_complete(self.stop())


# Example usage demonstrating both sync and async compatibility
def example():
    # Create a message bus
    bus = MessageBus()
    
    # Synchronous subscriber
    @bus.subscribe("data_updated")
    def on_data_updated(data):
        print(f"Sync handler received: {data}")
    
    # Asynchronous subscriber
    @bus.subscribe("data_updated")
    async def on_data_updated_async(data):
        await asyncio.sleep(0.1)  # Simulate async work
        print(f"Async handler received: {data}")
    
    # Synchronous data provider
    @bus.provide_data("config")
    def get_config():
        return {"version": "1.0", "mode": "test"}
    
    # Asynchronous data provider
    @bus.provide_data("user_info")
    async def get_user_info(user_id):
        await asyncio.sleep(0.1)  # Simulate database query
        return {"id": user_id, "name": f"User {user_id}"}
    
    # Sync loop method
    @bus.loop_method(delay=1.0)
    def check_sync():
        print("Sync loop check")
    
    # Async loop method
    @bus.loop_method(delay=1.5)
    async def check_async():
        await asyncio.sleep(0.1)
        print("Async loop check")
    
    # Run synchronously
    print("\n=== Synchronous API ===")
    # Get data from sync provider
    config = bus.get_data("config")
    print(f"Got config: {config}")
    
    # Get data from async provider
    user = bus.get_data("user_info", 42)
    print(f"Got user: {user}")
    
    # Publish event
    bus.publish("data_updated", {"source": "sync", "value": 100})
    
    # Allow time for async handlers to process
    import time
    time.sleep(0.2)
    
    # Run asynchronously
    async def async_demo():
        print("\n=== Asynchronous API ===")
        # Start loops
        await bus.start()
        
        # Get data
        config = await bus.get_data_async("config")
        print(f"Got config async: {config}")
        
        user = await bus.get_data_async("user_info", 43)
        print(f"Got user async: {user}")
        
        # Publish event
        await bus.publish_async("data_updated", {"source": "async", "value": 200})
        
        # Let it run for a bit
        await asyncio.sleep(3)
        
        # Stop loops
        await bus.stop()
    
    # Run the async demo
    asyncio.run(async_demo())


if __name__ == "__main__":
    example()