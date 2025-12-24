import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_id = "watt-ai/watt-tool-8B"
# model_id = "unsloth/Llama-3.2-1B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', device_map="auto", load_in_4bit=True)
# model = PeftModel.from_pretrained(model, "/home/pooh/code/utachi-ai-finetune-local/functioncall finetune/2bmodel/checkpoint-120",low_cpu_mem_usage=True)
# Example usage (adapt as needed for your specific tool usage scenario)
# system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
# If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out in a Json format.
# You should only return the function call in tools call sections.

# If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
# You SHOULD NOT include any other text in the response.
# Here is a list of functions in JSON format that you can invoke.\n{functions}\n
# """
# # User query
# query = "Find me the sales growth rate for company XYZ for the last 3 years and also the interest coverage ratio for the same duration."


# Initialize OpenAI client
# Set OPENAI_API_KEY env var instead
client = AsyncOpenAI(api_key="your-api-key-here",
                     base_url="http://127.0.0.1:11434/v1")

MODEL = "phi4-mini:3.8b"

system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only invoke the functions provided in the tools list."""

# User query
query = "Find me the sales growth rate for company XYZ for the last 3 years and also the interest coverage ratio for the same duration."


async def llm_functioncalling(query: str, tools: List[Dict[str, Any]]) -> str:
    """
    Asynchronously call OpenAI with function calling capabilities.

    Args:
        query: The user query text
        tools: List of tool definitions in OpenAI format

    Returns:
        Model response with function calls
    """
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': query}
    ]

    # Call OpenAI API with tools
    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
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
# async def llm_functioncalling(querry: str, function_list: str) -> str:
#     """
#     Asynchronously call the LLM with function calling capabilities.

#     Args:
#         querry: The user query text
#         function_list: JSON string containing function definitions

#     Returns:
#         Model response with function calls
#     """
#     messages = [
#         {'role': 'system', 'content': system_prompt.format(functions=function_list)},
#         {'role': 'user', 'content': querry}
#     ]

#     # Run tokenization and model generation in a thread pool to avoid blocking
#     loop = asyncio.get_event_loop()
#     with concurrent.futures.ThreadPoolExecutor() as pool:
#         # Tokenize input (potentially CPU-intensive)
#         inputs = await loop.run_in_executor(
#             pool,
#             lambda: tokenizer.apply_chat_template(
#                 messages,
#                 add_generation_prompt=True,
#                 return_tensors="pt"
#             ).to(model.device)
#         )

#         # Generate response (GPU-intensive)
#         outputs = await loop.run_in_executor(
#             pool,
#             lambda: model.generate(
#                 inputs,
#                 max_new_tokens=1024,
#                 do_sample=False,
#                 num_return_sequences=1,
#                 eos_token_id=tokenizer.eos_token_id
#             )
#         )

#         # Decode output (CPU-intensive)
#         results = await loop.run_in_executor(
#             pool,
#             lambda: tokenizer.decode(
#                 outputs[0][len(inputs[0]):],
#                 skip_special_tokens=True
#             )
#         )

#     print("model outputted: " + results)
#     return results
