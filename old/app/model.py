import re
from datetime import datetime, timedelta
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Union

# class Message:


class Command:
    def __init__(self,
                 name: str,
                 handler: Callable,
                 description: str = "",
                 patterns: Optional[List[str]] = None,
                 example_phrase: Optional[Dict[str, str]] = None,
                 required_entities: Optional[List[str]] = None,
                 dependency_rules: Optional[Dict] = None):
        """
        Command container with parsing logic

        :param name: Unique command identifier
        :param handler: Function to call when command is matched
        :param description: Help text for the command
        :param patterns: List of regex patterns to match
        :param required_entities: Required spaCy entity labels (e.g., ["TIME", "DATE"])
        :param dependency_rules: Dependency parsing rules (see example below)
        """
        self.name = name
        self.handler = handler
        self.description = description
        self.patterns = [re.compile(p, re.IGNORECASE)
                         for p in (patterns or [])]
        self.required_entities = required_entities or []
        self.dependency_rules = dependency_rules or {}
        self.example_phrase = example_phrase or {}

    def _extract_parameters(self) -> tuple[Dict[str, Dict], List[str]]:
        """Extract parameters from handler function signature."""
        import inspect

        signature = inspect.signature(self.handler)
        properties = {}
        required_params = []

        for param_name, param in signature.parameters.items():
            if param_name == "kwargs":
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

    def to_openai_format(self) -> dict:
        """
        Convert command to OpenAI function calling format.
        Compatible with OpenAI library and API.

        :return: Dictionary with OpenAI function calling schema
        """
        properties, required_params = self._extract_parameters()

        function_schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                }
            }
        }

        if required_params:
            function_schema["function"]["parameters"]["required"] = required_params

        return function_schema

    def to_ollama_format(self) -> dict:
        """
        Convert command to Ollama function calling format.

        :return: Dictionary with Ollama function calling schema
        """
        properties, required_params = self._extract_parameters()

        function_schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                }
            }
        }

        if required_params:
            function_schema["function"]["parameters"]["required"] = required_params

        return function_schema

    def to_claude_format(self) -> dict:
        """
        Convert command to Claude (Anthropic) function calling format.

        :return: Dictionary with Claude function calling schema
        """
        properties, required_params = self._extract_parameters()

        function_schema = {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required_params if required_params else []
            }
        }

        return function_schema
    # def to_function_calling_format(self) -> dict:
    #     """
    #     Convert command to LLM function calling format

    #     :return: Dictionary with function calling schema
    #     """
    #     import inspect

    #     # Get parameters from the handler function
    #     signature = inspect.signature(self.handler)
    #     parameters = {}
    #     required_params = []

    #     for param_name, param in signature.parameters.items():
    #         print(param_name)
    #         print(param)
    #         if param_name != "kwargs":

    #             param_type = "string"  # Default type
    #             param_description = ""

    #             # Extract parameter type annotation if available
    #             if param.annotation != inspect.Parameter.empty:
    #                 if param.annotation == str:
    #                     param_type = "string"
    #                 elif param.annotation == int:
    #                     param_type = "integer"
    #                 elif param.annotation == float:
    #                     param_type = "number"
    #                 elif param.annotation == bool:
    #                     param_type = "boolean"
    #                 elif param.annotation == list or param.annotation == List:
    #                     param_type = "array"
    #                 elif param.annotation == dict or param.annotation == Dict:
    #                     param_type = "object"

    #             # Add parameter to schema
    #             parameters[param_name] = {
    #                 "type": param_type,
    #                 "description": f"Parameter: {param_name}"
    #             }

    #         # Add required parameter if no default value
    #         if param.default == inspect.Parameter.empty:
    #             required_params.append(param_name)

    #     # Build function schema
    #     function_schema = {
    #         "name": self.name,
    #         "description": self.description,
    #         "parameters": {
    #             "type": "object",
    #             "properties": parameters,
    #         }
    #     }

    #     # Add examples if available
    #     if self.example_phrase:
    #         function_schema["examples"] = [
    #             {"role": "user", "content": phrase}
    #             for phrase in self.example_phrase.values()
    #         ]

    #     # Add required parameters if any
    #     if required_params:
    #         function_schema["parameters"]["required"] = required_params

    #     return function_schema
    # def to_claude_function_format(self) -> dict:
    #     """
    #     Convert command to Claude function calling format

    #     :return: Dictionary with Claude function calling schema
    #     """
    #     import inspect

    #     # Get parameters from the handler function
    #     signature = inspect.signature(self.handler)
    #     properties = {}
    #     required_params = []

    #     for param_name, param in signature.parameters.items():
    #         param_type = "string"  # Default type
    #         print(param_name)
    #         print(param)
    #         if param_name != "kwargs":
    #             # Extract parameter type annotation if available
    #             if param.annotation != inspect.Parameter.empty:
    #                 # Check for Optional type pattern (Union[X, None] or Optional[X])
    #                 origin = getattr(param.annotation, "__origin__", None)
    #                 args = getattr(param.annotation, "__args__", None)

    #                 # Handle Optional[X] or Union[X, None] pattern
    #                 is_optional = False
    #                 actual_type = param.annotation

    #                 if origin is Union and args and len(args) == 2 and type(None) in args:
    #                     is_optional = True
    #                     # Get the actual type (the one that isn't None)
    #                     actual_type = args[0] if args[1] is type(None) else args[1]

    #                 # Determine type based on actual_type
    #                 if actual_type == str:
    #                     param_type = "string"
    #                 elif actual_type == int:
    #                     param_type = "integer"
    #                 elif actual_type == float:
    #                     param_type = "number"
    #                 elif actual_type == bool:
    #                     param_type = "boolean"
    #                 elif actual_type == list or actual_type == List:
    #                     param_type = "array"
    #                 elif actual_type == dict or actual_type == Dict:
    #                     param_type = "object"

    #                 # Add parameter to schema
    #                 properties[param_name] = {
    #                     "type": param_type,
    #                     "description": f"{param_name.replace('_', ' ').capitalize()}"
    #                 }

    #                 # Only add to required_params if not optional and has no default value
    #                 if param.default == inspect.Parameter.empty and not is_optional:
    #                     required_params.append(param_name)

    #     # Build function schema in the requested format
    #     function_schema = {
    #         "name": self.name,
    #         "description": self.description,
    #         "arguments": {
    #             "type": "dict",
    #             "properties": properties,
    #             "required": required_params if required_params else []
    #         }
    #     }

    #     return function_schema


class Context:
    def __init__(self):
        self.history = []
        self.active_command: Optional[str] = None
        self.slots: Dict[str, Any] = {}
        self.created_at: datetime = datetime.now()
        self.expires_in: timedelta = timedelta(minutes=2)

    def is_valid(self) -> bool:
        return datetime.now() < self.created_at + self.expires_in


if __name__ == "__main__":
    # Sample handler function with various parameter types
    def sample_handler(text: str, count: int = 5, enabled: bool = True,
                       items: Optional[List[str]] = None, config: Optional[Dict] = None) -> str:
        """Sample handler function with various parameter types"""
        return f"Processed {text} {count} times with enabled={enabled}"

    # Create a Command object
    command = Command(
        name="sample_command",
        handler=sample_handler,
        description="A sample command for demonstration purposes",
        patterns=["process text", "handle input"],
        example_phrase={"example1": "Process this text",
                        "example2": "Handle this input"},
        required_entities=["TEXT"]
    )

    # Convert the Command object to function calling format
    function_schema = command.to_claude_function_format()

    # Print the function schema
    print("Function Calling Format Schema:")
    pprint(function_schema)
