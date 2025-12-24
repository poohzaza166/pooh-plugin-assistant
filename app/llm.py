import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from bus import MessageBus, Priority
from google import genai
from loader import PluginLoader
from openai import AsyncOpenAI
from plugin_base import Plugin


class IntentPlugin(Plugin):
    """
    Base class for intent handling plugins.

    These plugins listen for user queries and can respond to them.
    Supports function calling LLM integration via `to_function_calling_format()`.

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

    def _extract_parameters(self) -> tuple[Dict[str, Dict], List[str]]:
        """Extract parameters from handle_intent method signature."""
        signature = inspect.signature(self.handle_intent)
        properties = {}
        required_params = []

        for param_name, param in signature.parameters.items():
            if param_name in ("self", "utterance", "context", "initialize",
                              ):
                continue

            param_type = "string"
            is_optional = False
            actual_type = param.annotation

            # Handle type annotations
            if param.annotation != inspect.Parameter.empty:
                origin = getattr(param.annotation, "__origin__", None)
                args = getattr(param.annotation, "__args__", None)

                # Handle Optional[X] or Union[X, None]
                if origin is Union and args and len(args) == 2 and type(None) in args:
                    is_optional = True
                    actual_type = args[0] if args[1] is type(None) else args[1]

                # Determine type
                if actual_type == str:
                    param_type = "string"
                elif actual_type == int:
                    param_type = "integer"
                elif actual_type == float:
                    param_type = "number"
                elif actual_type == bool:
                    param_type = "boolean"
                elif actual_type == list or actual_type == List:
                    param_type = "array"
                elif actual_type == dict or actual_type == Dict:
                    param_type = "object"

            properties[param_name] = {
                "type": param_type,
                "description": f"{param_name.replace('_', ' ').capitalize()}"
            }

            if param.default == inspect.Parameter.empty and not is_optional:
                required_params.append(param_name)

        return properties, required_params

    def to_claude_format(self) -> Dict[str, Any]:
        """
        Convert intent handler to Claude (Anthropic) function calling format.

        Returns:
            Dictionary with Claude function calling schema
        """
        properties, required_params = self._extract_parameters()
        metadata = self.get_metadata()

        return {
            "name": metadata.name,
            "description": metadata.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "utterance": {
                        "type": "string",
                        "description": "User's spoken or typed input"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context information"
                    },
                    **properties
                },
                "required": ["utterance", "context"] + required_params
            }
        }

    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert intent handler to OpenAI function calling format.

        Returns:
            Dictionary with OpenAI function calling schema
        """
        properties, required_params = self._extract_parameters()
        metadata = self.get_metadata()

        return {
            "type": "function",
            "function": {
                "name": metadata.name,
                "description": metadata.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "utterance": {
                            "type": "string",
                            "description": "User's spoken or typed input"
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context information"
                        },
                        **properties
                    },
                    "required": ["utterance", "context"] + required_params
                }
            }
        }

    def to_ollama_format(self) -> Dict[str, Any]:
        """
        Convert intent handler to Ollama function calling format.

        Returns:
            Dictionary with Ollama function calling schema
        """
        properties, required_params = self._extract_parameters()
        metadata = self.get_metadata()

        return {
            "type": "function",
            "function": {
                "name": metadata.name,
                "description": metadata.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "utterance": {
                            "type": "string",
                            "description": "User's spoken or typed input"
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context information"
                        },
                        **properties
                    },
                    "required": ["utterance", "context"] + required_params
                }
            }
        }

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
    async def llm_run(
        self,
        utterance: str,
        context: Dict[str, Any],
        **kwargs: Any
    ) -> dict:
        """
        Run the intent plugin as a function-callable LLM endpoint.

        Args:
            utterance: The user's spoken/typed input
            context: Additional context information
            **kwargs: Additional parameters for the plugin

        Returns:
            dict: The plugin's response in a structured format
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


system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only invoke the functions provided in the tools list."""

MODEL = "phi4-mini:3.8b"


class LLM_Parser:
    def __init__(self, bus: MessageBus, pluginLoader: PluginLoader):
        self.bus = bus
        self.openai_client = AsyncOpenAI(api_key="",
                                         base_url="http://127.0.0.1:11434/v1")
        self.tools = []
        self.pluginLoader = pluginLoader
        self.register_tools()
        self.intent_responded_event = asyncio.Event()

    async def _infrence_llm(self, model_name: str, **kwargs):
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': kwargs.get("utterance")}
        ]

        # Call OpenAI API with tools
        response = await self.openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        # Extract tool calls from response
        result = ""
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                result += f"{tool_call.function.name}({tool_call.function.arguments})\n"
        else:
            result = response.choices[0].message.content

        print("model outputted: " + result)
        return result

    def register_tools(self):
        plugins = self.pluginLoader.get_all_plugin()
        for i in plugins:
            self.tools.append(i.to_claude_format())

    def _llm_parse_event(self):
        @self.bus.subscribe("intent.response", priority=Priority.HIGH)
        async def reset_respond(*args, **kwargs):
            self.intent_responded_event.set()

        @self.bus.subscribe("intent.query", priority=Priority.LOW)
        async def on_user_query(content, **kwargs):
            self.intent_responded_event.clear()
            try:
                # Wait for up to 3 seconds for a plugin to respond
                await asyncio.wait_for(self.intent_responded_event.wait(), timeout=3)
                # If we get here, a plugin responded in time
                logging.info("Plugin had Responded skipping LLM check")
                return
            except asyncio.TimeoutError:
                # No plugin responded in time, fallback to LLM
                logging.info("No plugin responded, falling back to LLM.")
                try:
                    results = await self._infrence_llm("", utterance=content.get("utterance", ""))
                    fn = self.parse_llm_function_call(results)
                    await self.execute_llm_plugin_function(fn["name"], fn["arguments"])
                except ValueError as e:
                    logging.error(f"LLM fallback failed: {e}")

    def parse_llm_function_call(self, llm_response: dict) -> dict:
        # Example for OpenAI function calling format
        if "function_call" in llm_response:
            func_name = llm_response["function_call"]["name"]
            arguments = llm_response["function_call"]["arguments"]
            return {"name": func_name, "arguments": arguments}
        # Add more parsing logic for other formats if needed
        raise ValueError("No function call found in LLM response")

    async def execute_llm_plugin_function(self, func_name, arguments):
        plugin = self.pluginLoader.get_plugin(func_name)
        if plugin == None:
            # Call llm_run with extracted arguments
            return await plugin.llm_run(**arguments)
        raise ValueError(f"No plugin found for function: {func_name}")


# Type hint for message bus to avoid circular imports
# When actually using, import from message_bus module
if TYPE_CHECKING:
    from bus import MessageBus
