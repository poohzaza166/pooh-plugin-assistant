import json
import re
import xml.etree.ElementTree as ET
from typing import Optional, Type

from pydantic import BaseModel, Field, ValidationError, validator


class BaseTool(BaseModel):
    # name: str
    # description: str


    def run(self, interest: str) -> str:
        """Use the tool."""
        if not interest:
            raise ValueError("Interest cannot be empty.")
        return "Recommended Book"

    async def async_run(self, interest: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BookRecommendation does not support async")

def convert_base_tool_to_prompt(tool_class: Type[BaseTool]) -> dict:
    """Converts a BaseTool class to a prompt."""
    # Get the input schema from the tool's run method
    input_schema = tool_class.model_fields
    # print(type(input_schema))
    # Get the output schema from the tool's run method
    output_schema = tool_class.__annotations__.get('return', None)

    # Create a dictionary of input parameters with their types
    # Initialize an empty dictionary for the parameters
    parameters = {}
    # Iterate over the items in the input schema
    for k in input_schema:
        # print(type(k))
        if k!= "name" and k != "description":
            test = tool_class.model_fields.get(k)
            print(test)
            # Add the parameter to the dictionary with its type and description
            if test.is_required() == True:
                parameters[k] = {"type": str(test.annotation).replace("<class '",'').replace("'>",""), "description": test.description}


    return {
        "name": tool_class.__name__,
        "description": tool_class.__doc__,
        "run": tool_class.run,
        "parameters": parameters,
        "output_schema": str(output_schema),
    }


def extract_function_calls(completion):
    # print("DEBUG: completion")
    # print(completion)
    if completion.startswith(":") == True:
        completion = completion.replace(":", '', 1)
        print("error handling")

    completion = completion.replace("'", '"')
    completion = completion.strip()
    pattern_multiple = r"(<multiplefunctions>(.*?)</multiplefunctions>)"
    pattern_single = r"(<functioncall>(.*?)</functioncall>)"

    pattern_multiple_new = r"<\|multiplefunctions\|>(.*?)<\|/multiplefunctions\|>"
    pattern_single_new = r"<\|functioncall\|>(.*?)<\|/functioncall\|>"

    match1 = re.search(pattern_multiple, completion, re.DOTALL)
    if match1:
        print("multiple function call match")
        multiplefn = match1.group(1)
        root = ET.fromstring(multiplefn)
        functions = root.findall("functioncall")
        return [json.loads(fn.text.strip()) for fn in functions]

    match2 = re.search(pattern_single, completion, re.DOTALL)
    if match2:
        print("single function call matched")
        singlefn = match2.group(1)
        root = ET.fromstring(singlefn)
        function = root.text.strip()
        return [json.loads(function)]

    pattern_multiple = f"({pattern_multiple}|{pattern_multiple_new})"
    pattern_single = f"({pattern_single}|{pattern_single_new})"
    
    match1 = re.search(pattern_multiple, completion, re.DOTALL)
    if match1:
        print("multiple function call match")
        multiplefn = match1.group(1)
        
        # Extract all individual function calls within the multiple function call
        individual_calls = re.findall(r"<functioncall>(.*?)</functioncall>", multiplefn) + \
                           re.findall(r"<\|functioncall\|>(.*?)<\|/functioncall\|>", multiplefn)
        
        return [json.loads(call.strip()) for call in individual_calls]

    match2 = re.search(pattern_single, completion, re.DOTALL)
    if match2:
        print("single function call matched")
        singlefn = match2.group(1)
        return [json.loads(singlefn.strip())]

    if re.search(r"^(<functioncall>)?$",completion, re.DOTALL):
        print("SDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
        print(completion)
        # raise FunctionCallSyntaxError()

    return False    

def parse_function_call(text):
    # Try to parse as JSON first
    try:
        data = json.loads(text)
        print("--------")
        print(data)
        if isinstance(data, dict) and 'function' in data:
            return parse_json_function_call(data['function'])
    except json.JSONDecodeError:
        # return 
        pass

    # If not JSON, try to extract function call using regex
    return parse_text_function_call(text)

def parse_json_function_call(function_call):
    if isinstance(function_call, str):
        try:
            function_call = json.loads(function_call)
        except json.JSONDecodeError as e:
            raise FunctionCallSyntaxError(e)

    name = function_call.get('name', '')
    arguments = function_call.get('arguments', '')

    if isinstance(arguments, str):
        try:
            arguments =  json.loads(arguments)
        except json.JSONDecodeError as e:
            raise FunctionCallSyntaxError(e)  # Keep arguments as string if it's not valid JSON

    return {
        'name': name,
        'arguments': arguments
    }

def parse_text_function_call(text):
    function_match = re.search(r'"functioncall"\s*:\s*({[^}]*})', text)

    if not function_match:
        return False

    function_content = function_match.group(1)
    try:
        function_call = json.loads(function_content)
        return parse_json_function_call(function_call)
    except json.JSONDecodeError:
        # If JSON parsing fails, try regex
        name_match = re.search(r'"function"\s*:\s*"([^"]*)"', function_content)
        args_match = re.search(r'"parameters"\s*:\s*([\'"])((?:(?!\1).)*)\1', function_content)

        name = name_match.group(1) if name_match else ''
        arguments = args_match.group(2) if args_match else ''

        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as e:
            # pass  # Keep arguments as string if it's not valid JSON
            raise FunctionCallSyntaxError(e)

        return {
            'function': name,
            'parameters': arguments
        }

    
def run_function_calls(functions, tools: BaseTool) -> str:
    if functions:
        for function in functions:
            for tool in tools:
                if tool.__name__ == function["name"]:
                    try:
                        res = tool(**function["arguments"])
                        return res.run(**function["arguments"]), True
                    except ValidationError as e:
                        return str(e.errors()), False
    return "No functions found in the completion."


def parse_message_block(message):
    # Extract the JSON part from the message
    json_match = re.search(r'```json\n(.*?)\n```', message, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
        # Parse the JSON string
        try:
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError as e:
            return f"Error parsing JSON: {e}"
    else:
        return "No JSON found in the message"