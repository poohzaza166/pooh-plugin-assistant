import random
from typing import Any, Dict, List, Optional

from bus import Priority
from plugin_base import IntentPlugin, PluginMetadata

# from llm import IntentPlugin


class Joker(IntentPlugin):
    """
    Joker class plugin that return joke
    """

    def __init__(self, bus, config):
        super().__init__(bus, config)
        self.joke_list = [
            "what make a game tick, a tock",
            "what do you do when a chicken cross a road"
        ]

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="joke",
            version="1.0.0",
            description="Joke Creation plugin",
            author="Your Name",
            dependencies=[]
        )

    async def initialize(self) -> None:
        await super().initialize()

        @self.bus.subscribe("intent.query", priority=Priority.NORMAL)
        async def on_query(data: Dict[str, Any]):
            utterance = data.get("utterance", "")
            context = data.get("context", {})

            response = await self.handle_intent(utterance, context)
            if response:
                await self.bus.publish_async("intent.response", {
                    "plugin": self.get_metadata().name,
                    "utterance": utterance,
                    "response": response
                })

    def get_intent_keywords(self) -> List[str]:
        return [
            "joke",
            "funny",
            "make me laugh",
            "tell me a joke"
        ]

    async def handle_intent(self, utterance: str, context: Dict[str, Any]) -> Optional[str]:
        utterance_lower = utterance.lower()
        if any(keyword in utterance_lower for keyword in self.get_intent_keywords()):
            return random.choice(self.joke_list)
        return None

    async def shutdown(self) -> None:
        await super().shutdown()
