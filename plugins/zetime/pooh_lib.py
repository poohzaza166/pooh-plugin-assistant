import inspect
import re
from datetime import datetime, timedelta
from pprint import pprint
from typing import (Any, Callable, Dict, List, Optional, Pattern, Protocol,
                    Union)


class Command:
    def __init__(self, 
                 name: str,
                 handler: Callable,
                 description: str = "",
                 patterns: Optional[List[str]] = None,
                 example_phrase: Optional[Dict[str,str]] = None,
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
        self.patterns = [re.compile(p, re.IGNORECASE) for p in (patterns or [])]
        self.required_entities = required_entities or []
        self.dependency_rules = dependency_rules or {}
        self.example_phrase = example_phrase or {}

    def to_function_calling_format(self) -> dict:
        """
        Convert command to LLM function calling format
        
        :return: Dictionary with function calling schema
        """
        import inspect

        # Get parameters from the handler function
        signature = inspect.signature(self.handler)
        parameters = {}
        required_params = []
        
        for param_name, param in signature.parameters.items():
            print(param_name)
            print(param)
            if param_name != "kwargs":

                param_type = "string"  # Default type
                param_description = ""
                
                # Extract parameter type annotation if available
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == str:
                        param_type = "string"
                    elif param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list or param.annotation == List:
                        param_type = "array"
                    elif param.annotation == dict or param.annotation == Dict:
                        param_type = "object"
                
                # Add parameter to schema
                parameters[param_name] = {
                    "type": param_type,
                    "description": f"Parameter: {param_name}"
                }
            
            # Add required parameter if no default value
            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)
        
        # Build function schema
        function_schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": parameters,
            }
        }
        
        # Add examples if available
        if self.example_phrase:
            function_schema["examples"] = [
                {"role": "user", "content": phrase}
                for phrase in self.example_phrase.values()
            ]
            
        # Add required parameters if any
        if required_params:
            function_schema["parameters"]["required"] = required_params
            
        return function_schema
    def to_claude_function_format(self) -> dict:
        """
        Convert command to Claude function calling format
        
        :return: Dictionary with Claude function calling schema
        """
        import inspect

        # Get parameters from the handler function
        signature = inspect.signature(self.handler)
        properties = {}
        required_params = []
        
        for param_name, param in signature.parameters.items():
            param_type = "string"  # Default type
            print(param_name)
            print(param)
            if param_name != "kwargs":
                # Extract parameter type annotation if available
                if param.annotation != inspect.Parameter.empty:
                    # Check for Optional type pattern (Union[X, None] or Optional[X])
                    origin = getattr(param.annotation, "__origin__", None)
                    args = getattr(param.annotation, "__args__", None)
                    
                    # Handle Optional[X] or Union[X, None] pattern
                    is_optional = False
                    actual_type = param.annotation
                    
                    if origin is Union and args and len(args) == 2 and type(None) in args:
                        is_optional = True
                        # Get the actual type (the one that isn't None)
                        actual_type = args[0] if args[1] is type(None) else args[1]
                        
                    # Determine type based on actual_type
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
                    
                    # Add parameter to schema
                    properties[param_name] = {
                        "type": param_type,
                        "description": f"{param_name.replace('_', ' ').capitalize()}"
                    }
                    
                    # Only add to required_params if not optional and has no default value
                    if param.default == inspect.Parameter.empty and not is_optional:
                        required_params.append(param_name)
        
        # Build function schema in the requested format
        function_schema = {
            "name": self.name,
            "description": self.description,
            "arguments": {
                "type": "dict",
                "properties": properties,
                "required": required_params if required_params else []
            }
        }
            
        return function_schema
    
class Context:
    def __init__(self):
        self.history = []
        self.active_command: Optional[str] = None
        self.slots: Dict[str, Any] = {}
        self.created_at: datetime = datetime.now()
        self.expires_in: timedelta = timedelta(minutes=2)
    
    def is_valid(self) -> bool:
        return datetime.now() < self.created_at + self.expires_in

from typing import Any, Dict, Optional, Protocol

import spacy


class CommandParserInterface(Protocol):
    """
    Interface protocol for natural language command parsing.
    
    This protocol defines the structure and methods expected from
    any command parser implementation, allowing for different parsers
    to be used interchangeably with proper type checking.
    """
    nlp: spacy.Language
    commands: Dict[str, Command]
    context: Optional[Context]
    fuzzy_threshold: int
    
    def register_command(self, command: Command) -> None:
        """
        Register a command with the parser.
        
        Args:
            command: The command object to register
        """
        ...
    
    def parse_input(self, text: str) -> Dict[str, Any]:
        """
        Parse natural language input and identify matching commands with parameters.
        
        Args:
            text: Natural language input text to parse
            
        Returns:
            A dictionary containing command match information, which may include:
            - "command": The name of the matched command
            - "handler": The function to execute
            - "parameters": Extracted parameters for the command
            - "confidence": Confidence score of the match (0.0-1.0)
            - "error": Error message if no match found
            - "context_prompt": Prompt for missing information if in context
        """
        ...
    
    def execute_command(self, text: str) -> Any:
        """
        Full execution pipeline: parse input text and execute the matched command.
        
        Args:
            text: Natural language input text
            
        Returns:
            The result of executing the command, or error information
        """
        ...
    
    def _get_dependency_info(self, doc) -> Dict:
        """
        Extract dependency relationships from a parsed document.
        
        Args:
            doc: A spaCy parsed document
            
        Returns:
            Dictionary of dependency information for tokens
        """
        ...
    
    def _check_dependency_rules(self, dependencies: Dict, rules: Dict) -> bool:
        """
        Verify if dependency structure matches command rules.
        
        Args:
            dependencies: Extracted dependencies from document
            rules: Rules to check against
            
        Returns:
            True if dependencies match rules, False otherwise
        """
        ...
    
    def _extract_parameters(self, doc, command: Command) -> Dict[str, Any]:
        """
        Extract parameters using NER and dependency parsing.
        
        Args:
            doc: A spaCy parsed document
            command: The command to extract parameters for
            
        Returns:
            Dictionary of extracted parameters
        """
        ...
    
    def _handle_context_continuation(self, text: str) -> Dict[str, Any]:
        """
        Fill missing slots in an ongoing context.
        
        Args:
            text: User input for context continuation
            
        Returns:
            Updated context information or completed command
        """
        ...
    
    def _validate_slots(self, command: Command) -> bool:
        """
        Check if all required slots are filled.
        
        Args:
            command: The command to validate slots for
            
        Returns:
            True if all required slots are filled, False otherwise
        """
        ...
    
    def _generate_slot_prompt(self, command: Command) -> str:
        """
        Generate natural language prompt for missing slots.
        
        Args:
            command: The command with missing slots
            
        Returns:
            A natural language prompt for the user
        """
        ...
    
    def _fuzzy_match(self, text: str) -> Optional[Command]:
        """
        Find best fuzzy match using command descriptions and samples.
        
        Args:
            text: Text to match against commands
            
        Returns:
            Matched command or None if no match found
        """
        ...
# Define the protocol for type checking.
class MessageBusInterface(Protocol):
    def subscribe(self, event_name: str) -> Callable: ...
    def publish(self, event_name: str, *args, **kwargs) -> None: ...
    def provide_data(self, data_name: str) -> Callable: ...
    def get_data(self, data_name: str, *args, **kwargs) -> Any: ...
    def loop_method(self, delay: float = 0) -> Callable: ...
    def push_event(self, event_name: str) -> Callable: ...
    def trigger_event(self, event_name: str, *args, **kwargs) -> None: ...
    # def register_command(self, command: Any) -> None: ... # Add register_command

class CommandInterface(Protocol):
    """Protocol defining the Command interface for type hinting"""
    name: str
    handler: Callable
    description: str
    patterns: List[Pattern]
    required_entities: List[str]
    dependency_rules: Dict
    example_phrase: Dict[str, str]
    
    def to_function_calling_format(self) -> dict: ...
    def to_claude_function_format(self) -> dict: ...

# -----------------------------------------------------------------------------
# Decorators that attach metadata to functions.
# -----------------------------------------------------------------------------

def subscribe(event_name: str):
    def decorator(func: Callable):
        setattr(func, '_subscribe_event', event_name)
        return func
    return decorator

def provide_data(data_name: str):
    def decorator(func: Callable):
        setattr(func, '_provide_data', data_name)
        return func
    return decorator

def loop_method(delay: float = 0):
    def decorator(func: Callable):
        setattr(func, '_loop_method', delay)
        return func
    return decorator

def push_event(event_name: str):
    def decorator(func: Callable):
        setattr(func, '_push_event', event_name)
        return func
    return decorator

def register_command(
    name: Optional[str] = None,
    description: Optional[str] = None,
    patterns: Optional[List[str]] = None,
    example_phrase: Optional[Dict[str, str]] = None,
    required_entities: Optional[List[str]] = None,
    dependency_rules: Optional[Dict] = None
):
    """
    Decorator to register a method as a command in the parser.
    
    Example:
        @register_command(
            patterns=[r"set (?:a|the) timer for (\d+) (?:minutes|hours)"],
            required_entities=["TIME"]
        )
        def set_timer(self, time: str):
            print(f"Timer set for {time}")
    """
    def decorator(func: Callable):
        # Store command registration details as attributes on the function
        setattr(func, '_is_command', True)
        setattr(func, '_command_name', name or func.__name__)
        setattr(func, '_command_description', description or inspect.getdoc(func) or "")
        setattr(func, '_command_patterns', patterns or [])
        setattr(func, '_command_example_phrase', example_phrase or {})
        setattr(func, '_command_required_entities', required_entities or [])
        setattr(func, '_command_dependency_rules', dependency_rules or {})
        return func
    return decorator

# -----------------------------------------------------------------------------
# PluginBase that registers decorated methods on initialization.
# -----------------------------------------------------------------------------

class PluginBase:
    def __init__(self, message_bus: MessageBusInterface, parser:CommandParserInterface ):
        self.message_bus = message_bus
        self.parser = parser
        self._register_decorated_methods()

    def _register_decorated_methods(self):
        # Iterate over functions defined on the class.
        for name, func in inspect.getmembers(self.__class__, predicate=inspect.isfunction):
            # Bind the function to the instance.
            bound_method = func.__get__(self, self.__class__)
            
            # Register if marked with a subscribe decorator.
            if hasattr(func, '_subscribe_event'):
                event_name = getattr(func, '_subscribe_event')
                self.message_bus.subscribe(event_name)(bound_method)
            
            # Register if marked with a provide_data decorator.
            if hasattr(func, '_provide_data'):
                data_name = getattr(func, '_provide_data')
                self.message_bus.provide_data(data_name)(bound_method)
            
            # Register if marked with a loop_method decorator.
            if hasattr(func, '_loop_method'):
                delay = getattr(func, '_loop_method')
                self.message_bus.loop_method(delay)(bound_method)
            
            # Register if marked with a push_event decorator.
            if hasattr(func, '_push_event'):
                event_name = getattr(func, '_push_event')
                self.message_bus.push_event(event_name)(bound_method)
            
            # Register if marked as a command
            if hasattr(func, '_is_command'):

                # Create a Command object using the stored attributes
                command = Command(
                    name=getattr(func, '_command_name'),
                    handler=bound_method,
                    description=getattr(func, '_command_description'),
                    patterns=getattr(func, '_command_patterns'),
                    example_phrase=getattr(func, '_command_example_phrase'),
                    required_entities=getattr(func, '_command_required_entities'),
                    dependency_rules=getattr(func, '_command_dependency_rules')
                )
                
                # Register the command via the message bus
            # if hasattr(self.message_bus, 'register_command'):
                self.message_bus.publish("registed_command", command.name)
            # if hasattr(self.message_bus, 'parser'):

                self.parser.register_command(command)
    
    # Instance methods to publish and trigger events.
    def publish(self, event_name: str, *args, **kwargs):
        self.message_bus.publish(event_name, *args, **kwargs)
    
    def trigger_event(self, event_name: str, *args, **kwargs):
        self.message_bus.trigger_event(event_name, *args, **kwargs)