from pprint import pprint

from google import genai
from google.genai import types

sys_instruct="You are a Naturnal Language OS API. Your job is to parse and understand the user instruction or call a function."
client = genai.Client(api_key="AIzaSyC6ERKaP-TJSkOQQI1gA5nHFjqys8Ho2FA")
model_config = types.GenerateContentConfig(system_instruction=sys_instruct,
                                           temperature=0.02,
                                           top_p=1,
                                           top_k=1,
                                           candidate_count=2,
                                           )
print(client.models.get(model='gemini-2.0-flash'))

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="hello there",
    config=model_config

)
pprint(response)
pprint(response.candidates)

