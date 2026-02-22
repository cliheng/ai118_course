import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import function_tools
import gradio as gr

load_dotenv()

Client = OpenAI(
    api_key=os.environ["ZHIPU_API_KEY"],
    base_url=os.environ["ZHIPU_API_BASE"]
)

tools = [
    function_tools.WEATHER_SEARCH
]


def weather_chat(city, platform):
    # 根据平台选择不同的Client和模型
    if platform == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_API_BASE")
        model = "gpt-3.5-turbo"
        client = OpenAI(api_key=api_key, base_url=base_url)
    elif platform == "zhipuai":
        api_key = os.environ.get("ZHIPU_API_KEY")
        base_url = os.environ.get("ZHIPU_API_BASE")
        model = "glm-4-flash"
        client = OpenAI(api_key=api_key, base_url=base_url)
    elif platform == "bailian":
        api_key = os.environ.get("BAILIAN_API_KEY")
        base_url = os.environ.get("BAILIAN_API_BASE")
        model = os.environ.get("BAILIAN_MODEL", "bailian-model")
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        return "请选择大模型平台。"

    messages=[
        {"role":"system","content":"不需要要求用户补充提问，必需使用工具tools返回的数据。根据数据回答用户询问的地点的天气情况，依据API查询当地的天气情况" },
        {"role":"user","content": city}
    ]
    response = client.chat.completions.create(
        model = model,
        messages=messages,
        tools =tools,
        tool_choice="auto"
    )
    print(f"############# {response}")
    while response.choices[0].message.tool_calls is not None:
        messages.append(response.choices[0].message)
        for tool_call in response.choices[0].message.tool_calls:
            args = tool_call.function.arguments
            args = json.loads(args)
            function_name = tool_call.function.name
            invoke_fun = getattr(function_tools,function_name)
            result = invoke_fun(**args)
            messages.append(
                {
                    "role":"tool",
                    "content":json.dumps(result),
                    "tool_call_id": tool_call.id
                }
            )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools
        )
    return response.choices[0].message.content

with gr.Blocks() as demo:
    gr.Markdown("# 天气查询助手")
    example_list = [
        ["今天北京天气怎么样？"],
        ["今天苏州天气怎么样？"],
        ["今天杭州天气怎么样？"]
    ]
    with gr.Row():
        platform = gr.Radio(
            choices=["openai", "zhipuai", "bailian"],
            value="zhipuai",
            label="大模型平台"
        )
    output = gr.Textbox(label="天气结果",lines=10)
    city = gr.Textbox(label="请输入城市名或天气问题", placeholder="如：北京天气")
    btn = gr.Button("查询")
    btn.click(fn=weather_chat, inputs=[city, platform], outputs=output)
    gr.Examples(
        examples=example_list,
        inputs=city
    )


if __name__ == "__main__":
    demo.launch()
