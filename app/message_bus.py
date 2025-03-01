
from typing import Any, Callable, Dict, List


class MessageBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.data_providers: Dict[str, Callable] = {}
        self.loop_methods: List[tuple] = []
        self.event_pushers: Dict[str, Callable] = {}

    def subscribe(self, event_name: str):
        def decorator(func: Callable):
            if event_name not in self.subscribers:
                self.subscribers[event_name] = []
            self.subscribers[event_name].append(func)
            return func
        return decorator

    def publish(self, event_name: str, *args, **kwargs):
        if event_name in self.subscribers:
            for subscriber in self.subscribers[event_name]:
                subscriber(*args, **kwargs)

    def provide_data(self, data_name: str):
        def decorator(func: Callable):
            self.data_providers[data_name] = func
            return func
        return decorator

    def get_data(self, data_name: str, *args, **kwargs) -> Any:
        if data_name in self.data_providers:
            return self.data_providers[data_name](*args, **kwargs)
        else:
            raise KeyError(f"No data provider found for '{data_name}'")

    def loop_method(self, delay: float = 0):
        print("i was called at some point")
        def decorator(func: Callable):
            self.loop_methods.append((func, delay))
            return func
        return decorator

    def push_event(self, event_name: str):
        def decorator(func: Callable):
            self.event_pushers[event_name] = func
            return func
        return decorator

    def trigger_event(self, event_name: str, *args, **kwargs):
        if event_name in self.event_pushers:
            result = self.event_pushers[event_name](*args, **kwargs)
            self.publish(event_name, result)
        else:
            print(f"No event pusher found for '{event_name}'")


