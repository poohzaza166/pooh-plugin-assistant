#!/usr/bin/env python3
"""
AI Voice Assistant Main Application
"""
import asyncio
import logging

from bus import MessageBus, Priority
from llm import LLM_Parser
from loader import PluginLoader
from memory import History
from mic_util import WakeWordWhisper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class VoiceAssistant:
    """Main voice assistant application."""

    def __init__(self, plugins_dir: str = "plugins"):
        self.bus = MessageBus()
        self.plugin_loader = PluginLoader(self.bus, plugins_dir)
        self.logger = logging.getLogger("voice_assistant")
        self.history = History(self.bus)
        self.llm_parser = LLM_Parser(self.bus, self.plugin_loader)
        self._setup_core_events()

    def _setup_core_events(self) -> None:
        """Set up core event handlers."""

        self.history._core_events()
        # self.llm_parser._llm_parse_event()

        @self.bus.subscribe("intent.response", priority=Priority.CRITICAL)
        async def on_intent_response(data: dict):
            """Handle intent responses from plugins."""
            plugin_name = data.get("plugin", "unknown")
            response = data.get("response", "")
            self.logger.info(f"[{plugin_name}] Response: {response}")

        @self.bus.subscribe("system.error", priority=Priority.CRITICAL)
        async def on_error(data: dict):
            """Handle system errors."""
            error = data.get("error", "Unknown error")
            self.logger.error(f"System error: {error}")

        @self.bus.loop_method(delay=10.0)
        async def health_check():
            """Periodic health check."""
            loaded = len(self.plugin_loader.loaded_plugins)
            self.logger.debug(f"Health check: {loaded} plugins loaded")

    async def start(self) -> None:
        """Start the voice assistant."""
        self.logger.info("Starting voice assistant...")

        # Load all plugins
        loaded_count = await self.plugin_loader.load_all_plugins()
        self.logger.info(f"Loaded {loaded_count} plugins")

        # List loaded plugins
        for metadata in self.plugin_loader.list_plugins():
            self.logger.info(
                f"  - {metadata.name} v{metadata.version} by {metadata.author}"
            )

        # Start the message bus (for loop methods)
        await self.bus.start()

        self.logger.info("Voice assistant started successfully")

    async def stop(self) -> None:
        """Stop the voice assistant."""
        self.logger.info("Stopping voice assistant...")

        # Stop the message bus
        await self.bus.stop()

        # Unload all plugins
        await self.plugin_loader.unload_all_plugins()

        self.logger.info("Voice assistant stopped")

    async def process_query(self, utterance: str, context: dict = None) -> None:
        """
        Process a user query.

        Args:
            utterance: User's input text
            context: Optional context dictionary
        """
        if context is None:
            context = {}

        self.logger.info(f"Processing query: {utterance}")

        # Publish intent query event
        await self.bus.publish_async("intent.query", {
            "utterance": utterance,
            "context": context
        })

    async def get_bus(self) -> MessageBus:
        if self.bus == None:
            raise ValueError("the bus is not init some how")
        return self.bus


async def main():
    """Main entry point."""
    assistant = VoiceAssistant(plugins_dir="plugins")
    bus = await assistant.get_bus()
    wake_word = WakeWordWhisper(bus=bus,
                                vosk_model_path="/mnt/driveD/code/python/le-pooh-assistant/vosk-model-small-en-us-0.15")
    wake_word.start()

    try:
        # Start the assistant
        await assistant.start()

        # Simulate some queries
        await asyncio.sleep(1)
        while True:
            await asyncio.sleep(999999)
            pass

        # print("\n" + "="*60)
        # print("Testing Weather Plugin")
        # print("="*60 + "\n")

        # await assistant.process_query("What's the weather like?")
        # await asyncio.sleep(0.5)

        # await assistant.process_query("How hot is it?")
        # await asyncio.sleep(0.5)

        # # Should not be handled
        # await assistant.process_query("Tell me a joke")
        # await asyncio.sleep(0.5)

        # await assistant.process_query("what is the current time")
        # await asyncio.sleep(0.5)

        # # Let it run for a bit to see health checks
        # print("\nRunning for 15 seconds...")
        # await asyncio.sleep(15)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await assistant.stop()


if __name__ == "__main__":
    asyncio.run(main())
