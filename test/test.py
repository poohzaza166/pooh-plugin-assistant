from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "watt-ai/watt-tool-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto', device_map="auto")

# Example usage (adapt as needed for your specific tool usage scenario)
system_prompt="""You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.\n{functions}\n
"""
# User query
query = "Find me the sales growth rate for company XYZ for the last 3 years and also the interest coverage ratio for the same duration."

tools = [
    {
        "name": "financial_ratios.interest_coverage", "description": "Calculate a company's interest coverage ratio given the company name and duration",
        "arguments": {
            "type": "dict",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "The name of the company."
                }, 
                "years": {
                    "type": "integer",
                    "description": "Number of past years to calculate the ratio."
                }
            }, 
            "required": ["company_name", "years"]
        }
    },
    {
        "name": "sales_growth.calculate",
        "description": "Calculate a company's sales growth rate given the company name and duration",
        "arguments": {
            "type": "dict", 
            "properties": {
                "company": {
                    "type": "string",
                    "description": "The company that you want to get the sales growth rate for."
                }, 
                "years": {
                    "type": "integer",
                    "description": "Number of past years for which to calculate the sales growth rate."
                }
            }, 
            "required": ["company", "years"]
        }
    },
    {
        "name": "weather_forecast",
        "description": "Retrieve a weather forecast for a specific location and time frame.",
        "arguments": {
            "type": "dict",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city that you want to get the weather for."
                }, 
                "days": {
                    "type": "integer",
                    "description": "Number of days for the forecast."
                }
            },
            "required": ["location", "days"]
        }
    }
]

messages = [
    {'role': 'system', 'content': system_prompt.format(functions=tools)},
    {'role': 'user', 'content': query}
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
