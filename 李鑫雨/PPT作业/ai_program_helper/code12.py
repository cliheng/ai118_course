import os
from zhipuai import OpenAI
from dotenv import load_dotenv, find_dotenv
def llm_chat(message):
    client=zhipuai(api_key=os.environ['0082a06d697f4e908b58953daec13d3a.sVbjsuV02Je3lebu'])
    response = client.chat.completions.create(
        model="glm-4-flash-250414",
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message
    return result
