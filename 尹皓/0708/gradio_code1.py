import gradio as gr
from openai import OpenAI
import os
import json
import functioncall_tools
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"]
)

def chatbot_interface(query):
    tools = [functioncall_tools.WEATHER_SEARCH]
    
    messages = [
        {"role": "system", "content": "不需要要求用户补充问题,直接按问题调用tool"},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    while response.choices[0].message.tool_calls is not None:
        messages.append(response.choices[0].message)
        
        for tool_call in response.choices[0].message.tool_calls:
            if tool_call.type == "function":
                args = tool_call.function.arguments
                args = json.loads(args)
                function_name = tool_call.function.name
                invoke_fun = getattr(functioncall_tools, function_name)
                result = invoke_fun(**args)
                
                messages.append({
                    "role": "tool",
                    "content": f"{json.dumps(result)}",
                    "tool_call_id": tool_call.id,
                })
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    tools=tools,
                )
    
    return response.choices[0].message.content

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# 实战：天气查询助手(Function Calling)")
    
    with gr.Row():
        with gr.Column(scale=2):
            query = gr.Textbox(label="请输入查询内容", lines=6)
            submit_button = gr.Button("提交", size="lg")
        
        with gr.Column(scale=1):
            text_output = gr.Textbox(label="查询结果", lines=10)
    
    submit_button.click(
        fn=chatbot_interface,
        inputs=query,
        outputs=text_output
    )
    
    examples = [
        ["今天北京天气如何？"],
        ["上海现在的天气怎么样？"],
        ["广州未来几天的天气情况"]
    ]
    gr.Examples(examples=examples, inputs=query)

demo.launch()