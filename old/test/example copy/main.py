import asyncio
import threading
import time
from datetime import datetime

from .pooh_lib import (PluginBase, loop_method, provide_data, push_event,
                       register_command, subscribe)


def plugin_entry(message_bus, parser):
    print("loading plugin")
    print("checking parser object")
    print(parser)
    plugin = ExamplePlugin(message_bus, parser)
    return  plugin

class ExamplePlugin(PluginBase):
    def __init__(self, message_bus,parser):
        print(parser)
        super().__init__(message_bus, parser)

    @subscribe("example_event")
    def handle_example_event(self, message):
        print(f"Example plugin received: {message}")

    @provide_data("example_data")
    def provide_example_data(self, param):
        return f"Example data with param: {param}"

    @loop_method(delay=1)
    async def async_task(self):
        print(f"Async task running... (Thread: {threading.current_thread().name})")
        await asyncio.sleep(0.5)

    @loop_method(delay=2)
    def sync_task(self):
        print(f"Sync task running... (Thread: {threading.current_thread().name})")
        time.sleep(1)

    @subscribe("text_input")
    def say_hi(self, message):
        self.parser.execute_command(message)

    async def run(self):
        print("Example plugin is running")
        while True:
            await asyncio.sleep(2)
            # print("Example plugin still active")

    @register_command(example_phrase = ["whats the current time"])
    def get_current_time(self):
        print(datetime.now().time())

    @register_command(
        patterns=[r"hello (bot|assistant)"],
        example_phrase={"greeting": "Hello assistant"}
    )
    def hello_command(self, target: str = "bot"):
        """Respond to a greeting with a friendly message"""
        return f"Hello! I'm your {target}. How can I help you today?"
    