from typing import Any, Dict, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "watt-ai/watt-tool-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', device_map="auto")

# Example usage (adapt as needed for your specific tool usage scenario)
system_prompt="""You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out in a Json format.
You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.\n{functions}\n
"""
# User query
query = "Find me the sales growth rate for company XYZ for the last 3 years and also the interest coverage ratio for the same duration."

def llm_functioncalling(querry: str, function_list: str) -> str:
    # print()
    messages = [
        {'role': 'system', 'content': system_prompt.format(functions=function_list)},
        {'role': 'user', 'content': querry}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    outputs = model.generate(inputs, max_new_tokens=1024, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    results = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    print("model outputted" + results)
    return results