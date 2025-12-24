"""
Plugin Base Classes

This module defines the base classes that all plugins should inherit from.
Plugins can import these classes to get full IDE autocomplete support.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)


class Plugin(ABC):
    """
    Base class for all plugins.

    Plugins should inherit from this class and implement the required methods.
    The message bus is available via self.bus for subscribing to events.

    Example:
        class MyPlugin(Plugin):
            def get_metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="my_plugin",
                    version="1.0.0",
                    description="My awesome plugin",
                    author="Your Name"
                )

            async def initialize(self) -> None:
                await super().initialize()

                @self.bus.subscribe("some.event")
                async def on_event(data):
                    self.logger.info(f"Got event: {data}")
    """

    def __init__(self, bus: 'MessageBus', config: Dict[str, Any]):
        """
        Initialize the plugin.

        Args:
            bus: The message bus instance
            config: Plugin configuration from YAML
        """
        self.bus = bus
        self.config = config
        self.logger = logging.getLogger(f"plugin.{self.get_metadata().name}")
        self._initialized = False

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """
        Return metadata about this plugin.

        Returns:
            PluginMetadata object with plugin information
        """
        pass

    async def initialize(self) -> None:
        """
        Initialize the plugin. Called when the plugin is loaded.
        Override this to set up your plugin, subscribe to events, etc.

        Example:
            async def initialize(self) -> None:
                await super().initialize()

                # Subscribe to events
                @self.bus.subscribe("user.query")
                async def handle_query(data):
                    query = data.get("query", "")
                    self.logger.info(f"Got query: {query}")

                # Register data providers
                @self.bus.provide_data("my_plugin.data")
                def get_data():
                    return {"key": "value"}
        """
        self._initialized = True
        self.logger.info(f"Plugin {self.get_metadata().name} initialized")

    async def shutdown(self) -> None:
        """
        Shutdown the plugin. Called when the plugin is being unloaded.
        Override this to clean up resources.

        Example:
            async def shutdown(self) -> None:
                # Clean up resources
                self.logger.info("Cleaning up...")
                await super().shutdown()
        """
        self._initialized = False
        self.logger.info(f"Plugin {self.get_metadata().name} shutdown")

    def is_initialized(self) -> bool:
        """Check if the plugin is initialized."""
        return self._initialized

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Supports dot notation for nested values.

        Args:
            key: Configuration key (supports dot notation for nested values)
            default: Default value if key is not found

        Returns:
            Configuration value or default

        Example:
            # Simple key
            api_key = self.get_config_value("api_key", "")

            # Nested key
            timeout = self.get_config_value("network.timeout", 30)
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default


class IntentPlugin(Plugin):
    """
    Base class for intent handling plugins.

    These plugins listen for user queries and can respond to them.

    Example:
        class WeatherPlugin(IntentPlugin):
            def get_metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="weather",
                    version="1.0.0",
                    description="Weather information",
                    author="Your Name"
                )

            def get_intent_keywords(self) -> List[str]:
                return ["weather", "temperature", "forecast"]

            async def handle_intent(self, utterance: str, context: Dict[str, Any]) -> Optional[str]:
                if "weather" in utterance.lower():
                    return "It's sunny and 72 degrees!"
                return None
    """

    @abstractmethod
    def get_intent_keywords(self) -> List[str]:
        """
        Return keywords or phrases this plugin can handle.

        These keywords are used to determine if this plugin should
        process a user's query.

        Returns:
            List of keywords/phrases

        Example:
            def get_intent_keywords(self) -> List[str]:
                return [
                    "weather",
                    "temperature",
                    "forecast",
                    "how hot",
                    "how cold"
                ]
        """
        pass

    @abstractmethod
    async def handle_intent(self, utterance: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Handle a user utterance.

        This method is called when a user query matches your keywords.
        Return a response string if you handle the intent, or None if not.

        Args:
            utterance: The user's spoken/typed input
            context: Additional context information (location, user prefs, etc.)

        Returns:
            Response text or None if intent not handled

        Example:
            async def handle_intent(self, utterance: str, context: Dict[str, Any]) -> Optional[str]:
                utterance_lower = utterance.lower()

                if "weather" in utterance_lower:
                    location = context.get("location", "your area")
                    # Get weather data
                    weather = await self.bus.get_data_async("weather.current", location)
                    return f"The weather in {location} is {weather['conditions']}"

                return None
        """
        pass


class VoicePlugin(Plugin):
    """
    Base class for voice-related plugins (STT, TTS, Wake Word, etc.).

    Example:
        class MyTTSPlugin(VoicePlugin):
            def get_metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="my_tts",
                    version="1.0.0",
                    description="Text-to-speech plugin",
                    author="Your Name"
                )

            async def process_audio(self, audio_data: bytes) -> str:
                # Convert text to speech
                return await self.synthesize_speech(audio_data)
    """

    @abstractmethod
    async def process_audio(self, audio_data: bytes) -> Any:
        """
        Process audio data.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Processed result (text for STT, audio for TTS, etc.)
        """
        pass

# Type hint for message bus to avoid circular imports
# When actually using, import from message_bus module


if TYPE_CHECKING:
    from message_bus import MessageBus
