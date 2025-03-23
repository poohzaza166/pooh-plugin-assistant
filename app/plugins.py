import asyncio
import importlib
import inspect
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI

from .message_bus import MessageBus
from .parser import UnifiedCommandParser


class PluginManager:

    def __init__(self, plugin_folder: str, parser: UnifiedCommandParser):
        self.plugin_folder = plugin_folder
        self.plugins = {}
        self.message_bus = MessageBus()
        
        self.parser = parser
        # self.api = api
        self.executor = ThreadPoolExecutor(max_workers=10)

    @staticmethod
    def get_bus():
        # if isinstance(cls.message_bus, MessageBus):
        return PluginManager.message_bus

    def load_plugins(self):
        for item in os.listdir(self.plugin_folder):
            plugin_path = os.path.join(self.plugin_folder, item)
            if os.path.isdir(plugin_path):
                self._load_plugin(item)

    def _load_plugin(self, plugin_name: str):
        # try:
            # Add plugin folder to Python path if not already there
            plugin_base_path = os.path.abspath(self.plugin_folder)
            if plugin_base_path not in sys.path:
                sys.path.insert(0, plugin_base_path)

            # Import the plugin package first (triggers __init__.py)
            package = importlib.import_module(plugin_name)
            
            # Then import the main module from the package
            plugin_module = importlib.import_module(f"{plugin_name}.main")
            
            # Look for entry point
            if hasattr(plugin_module, 'plugin_entry'):
                plugin_instance = plugin_module.plugin_entry(self.message_bus, self.parser)
                self.plugins[plugin_name] = plugin_instance
                print(f"Loaded plugin: {plugin_name}")
            else:
                print(f"Error: Plugin {plugin_name} does not have a plugin_entry function")
                
        # except Exception as e:
        #     print(f"Error loading plugin {plugin_name}: {str(e)}")
            
        # finally:
        #     # Clean up sys.path
        #     if plugin_base_path in sys.path:
        #         sys.path.remove(plugin_base_path)

    async def run_plugins(self):
        tasks = []
        for plugin_name, plugin_instance in self.plugins.items():
            print(f"founded {plugin_name}")
            if hasattr(plugin_instance, "run"):
                tasks.append(asyncio.create_task(plugin_instance.run()))
            else:
                print(f"Warning: Plugin {plugin_name} does not have a run method")
        
        for method, delay in self.message_bus.loop_methods:
            print("adding looped task")
            tasks.append(asyncio.create_task(self._run_loop_method_async(method, delay)))

        await asyncio.gather(*tasks)

    async def _run_loop_method_async(self, method, delay):
        loop = asyncio.get_event_loop()
        while True:
            await loop.run_in_executor(self.executor, self._run_method_in_thread, method)
            await asyncio.sleep(delay)

    def _run_method_in_thread(self, method):
        try:
            if asyncio.iscoroutinefunction(method):
                asyncio.run(method())
            else:
                method()
        except Exception as e:
            print(f"Error in loop method: {str(e)}")
            print(f"Thread: {threading.current_thread().name}")
