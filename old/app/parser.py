import asyncio
import inspect
import json
import re
from typing import Any, Callable, Dict, Optional

import spacy
from fuzzywuzzy import fuzz, process

from .function_call import llm_functioncalling
from .model import Command, Context


class UnifiedCommandParser:
    """
    A parser for handling and executing natural language commands using NLP.
    Attributes:
        nlp (spacy.Language): The spaCy language model for NLP tasks.
        commands (Dict[str, Command]): A dictionary of registered commands.
        context (Optional[Context]): The current context for ongoing command execution.
        fuzzy_threshold (int): The threshold for fuzzy matching similarity score (0-100).
    Methods:
        register_command(command: Command):
            Registers a new command to the parser.
        parse_input(text: str) -> Dict:
            Parses the input text and returns the matched command with parameters.
        execute_command(text: str):
            Parses the input text and executes the matched command.
        _get_dependency_info(doc) -> Dict:
            Extracts dependency relationships from the parsed document.
        _check_dependency_rules(dependencies: Dict, rules: Dict) -> bool:
            Verifies if the dependency structure matches the command rules.
        _extract_parameters(doc, command: Command) -> Dict:
            Extracts parameters using named entity recognition (NER) and dependency parsing.
        _handle_context_continuation(text: str) -> Dict:
            Handles filling missing slots in an ongoing context.
        _validate_slots(command: Command) -> bool:
            Checks if all required slots are filled for a command.
        _generate_slot_prompt(command: Command) -> str:
            Generates a natural language prompt for missing slots.
        _fuzzy_match(text: str) -> Optional[Command]:
            Finds the best fuzzy match using command descriptions and samples.
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.commands: Dict[str, Command] = {}
        self.context: Optional[Context] = None
        self.fuzzy_threshold = 75  # 0-100 similarity score

    def register_command(self, command: Command):
        self.commands[command.name] = command

    def parse_input(self, text: str) -> Dict:
        """Parse text and return matched command with parameters"""
        # First check if we're in a context continuation
        # if self.context and self.context.is_valid():
        #     return self._handle_context_continuation(text)

        # # Proceed with standard parsing
        # doc = self.nlp(text)

        # # Extract entities and dependencies
        # entities = {ent.label_: ent.text for ent in doc.ents}
        # dependencies = self._get_dependency_info(doc)

        # # Score commands based on match probability
        # candidates = []
        # for cmd in self.commands.values():
        #     score = 0

        #     # Check regex patterns
        #     for pattern in cmd.patterns:
        #         if pattern.search(text):
        #             score += 1
        #             break

        #     # Check required entities
        #     if all(e in entities for e in cmd.required_entities):
        #         score += len(cmd.required_entities)

        #     # Check dependency rules
        #     if self._check_dependency_rules(dependencies, cmd.dependency_rules):
        #         score += 1

        #     if score >= 3:
        #         print(score)
        #         candidates.append((cmd, score))

        # if candidates:
        #     # Get best match (could be extended with confidence thresholds)
        #     best_match = max(candidates, key=lambda x: x[1])
        #     # print(best_match)
        #     params = self._extract_parameters(doc, best_match[0])

        #     return {
        #         "command": best_match[0].name,
        #         "handler": best_match[0].handler,
        #         "parameters": params,
        #         # Normalize score
        #         "confidence": best_match[1] / (len(best_match[0].__dict__) - 2)
        #     }

        # fallback to LLM
        functioncall_str = []
        for cmd in self.commands.values():
            functioncall_str.append(cmd.to_openai_format())
        result = asyncio.run(llm_functioncalling(
            query=text, tools=functioncall_str))
        print("--"*10)
        print(result)
        print("#"*10)
        # if result != "":
        pattern = r"([\w\.]+)\(([^)]*)\)"

        matches = re.findall(pattern, result)
        if len(matches) > 0:
            # commands = []
            for match in matches:
                func_name, args = match
                params = {}

                # Parsing key-value arguments
                for param in args.split(","):
                    key_value = param.strip().split("=")
                    if len(key_value) == 2:
                        key, value = key_value
                        value = value.strip().strip("\"")  # Remove potential quotes
                        params[key] = value

            return {
                "command": func_name,
                # Assuming the function name is the handler
                "handler": self.commands[func_name].handler,
                "parameters": params,
                "confidence": 1  # Placeholder for confidence calculation
            }

        return {"error": "No matching command found"}

    def _get_dependency_info(self, doc) -> Dict:
        """Extract dependency relationships"""
        deps = {}
        for token in doc:
            deps[token.text.lower()] = {
                "dep": token.dep_,
                "head": token.head.text,
                "children": [child.text for child in token.children]
            }
        return deps

    def _check_dependency_rules(self, dependencies: Dict, rules: Dict) -> bool:
        """Verify if dependency structure matches command rules"""
        for token, conditions in rules.items():
            if token not in dependencies:
                return False
            for key, value in conditions.items():
                if dependencies[token].get(key) != value:
                    return False
        return True

    def _extract_parameters(self, doc, command: Command) -> Dict:
        """Extract parameters using NER and dependency parsing"""
        params = {}

        # Extract entities
        entities = {ent.label_: ent.text for ent in doc.ents}
        for ent in command.required_entities:
            if ent in entities:
                params[ent.lower()] = entities[ent]

        # Extract direct objects for verbs
        for token in doc:
            if token.dep_ == "dobj":
                params["target"] = token.text
                params["action"] = token.head.text

        return params

    def execute_command(self, text: str):
        """Full pipeline: parse + execute"""
        result = self.parse_input(text)
        # print(result)
        if result["confidence"] < 0.4:
            print("error")
            return result
        if result.get("error") == None:
            print("error")
            return result
        # print(result)

        output = execute_with_matched_params(
            result["handler"], result["parameters"])
        return output

    def _handle_context_continuation(self, text: str) -> Dict:
        """Fill missing slots in an ongoing context"""
        current_cmd = self.commands[self.context.active_command]

        # Extract parameters using multiple strategies
        params = self._extract_parameters(self.nlp(text), current_cmd)

        # Update context slots
        self.context.slots.update({k: v for k, v in params.items() if v})

        # Check if all required slots are filled
        if self._validate_slots(current_cmd):
            result = {
                "command": current_cmd.name,
                "handler": current_cmd.handler,
                "parameters": self.context.slots,
                "context_complete": True
            }
            self.context = None  # Clear context after completion
            return result
        else:
            return {
                "context_prompt": self._generate_slot_prompt(current_cmd),
                "context": self.context
            }

    def _validate_slots(self, command: Command) -> bool:
        """Check if all required slots are filled"""
        if not command.required_entities:
            return True
        return all(slot in self.context.slots for slot in command.required_entities)

    def _generate_slot_prompt(self, command: Command) -> str:
        """Generate natural language prompt for missing slots"""
        missing = [slot for slot in command.required_entities
                   if slot not in self.context.slots]

        prompts = {
            "TIME": "For how long?",
            "DATE": "When should I set this?",
            "LOCATION": "Where would you like this set?",
            "DEFAULT": "What else do you need?"
        }
        return next((prompts[slot] for slot in missing if slot in prompts),
                    prompts["DEFAULT"])

    def _fuzzy_match(self, text: str) -> Optional[Command]:
        """Find best fuzzy match using command descriptions and samples"""
        # Create search pool from command metadata
        search_pool = []
        for cmd in self.commands.values():
            search_pool.append(cmd.description)
            search_pool.extend(cmd.patterns)

        # Find best match using partial ratio
        best_match, score, _ = process.extractOne(
            text,
            search_pool,
            scorer=fuzz.partial_ratio
        )

        if score >= self.fuzzy_threshold:
            # Find which command the match belongs to
            for cmd in self.commands.values():
                if best_match in cmd.patterns or best_match == cmd.description:
                    return cmd
        return None


def execute_with_matched_params(handler: Callable, parameters: Dict[str, Any]) -> Any:
    """
    Execute a handler function with only parameters that match its signature.

    Args:
        handler: The function to execute
        parameters: Dictionary of parameters

    Returns:
        The result of the handler function

    Extra parameters not in the function signature will be logged to the console.
    """
    # Get the function signature
    signature = inspect.signature(handler)
    param_names = set(signature.parameters.keys())

    # Filter parameters that match function signature
    matched_params = {}
    extra_params = {}

    for name, value in parameters.items():
        if name in param_names:
            matched_params[name] = value
        else:
            extra_params[name] = value

    # Log extra parameters
    if extra_params:
        # logging.info(f"Extra parameters not used by {handler.__name__}: {extra_params}")
        print(
            f"[INFO] Extra parameters not used by {handler.__name__}: {extra_params}")

    # Check for missing required parameters
    missing_params = []
    for name, param in signature.parameters.items():
        if (param.default == inspect.Parameter.empty and
            name not in matched_params and
                param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)):
            missing_params.append(name)

    if missing_params:
        missing_msg = f"Missing required parameters for {handler.__name__}: {missing_params}"
        # logging.error(missing_msg)
        print(f"[ERROR] {missing_msg}")
        # Could raise an exception here, but we'll continue with what we have

    # Execute the handler with matched parameters
    return handler(**matched_params)
