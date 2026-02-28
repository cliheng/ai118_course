import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import function_tools
from zhipuai import ZhipuAI

def chatbot_interface(query, model_input, temperature):
    response = ''
    # 选择模型
    try:
        load_dotenv()
        zhipuai_client = ZhipuAI(
            api_key=os.environ.get("ZHIPU_API_KEY", ""),
            base_url=os.environ.get("ZHIPU_API_BASE", "")
        )
        openai_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_API_BASE", "")
        )
        model_map = {
            "openai": (openai_client, "gpt-3.5-turbo"),
            "ZhipuAI": (zhipuai_client, "glm-4-flash"),
            "Bailian": (openai_client, "gpt-3.5-turbo") # 占位
        }
        if model_input not in model_map:
            return "暂不支持该模型平台"
        client, model_name = model_map[model_input]
        tools = [function_tools.TIANQI_SEARCH]
        messages = [
            {'role': 'system', 'content': "不需要要求用户补充问题，直接按问题调用tool"},
            {"role": "user", "content": query}
        ]
        response_obj = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature
        )
        while response_obj.choices[0].message.tool_calls is not None:
            message_dict = response_obj.choices[0].message.model_dump()
            messages.append(message_dict)
            for tool_call in response_obj.choices[0].message.tool_calls:
                args = tool_call.function.arguments
                args = json.loads(args)
                fuction_name = tool_call.function.name
                invoke_fun = getattr(function_tools, fuction_name)
                result = invoke_fun(**args)
                messages.append(
                    {
                        "role": "tool",
                        "content": f"{json.dumps(result)}",
                        "tool_call_id": tool_call.id
                    }
                )
            response_obj = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                temperature=temperature
            )
        return response_obj.choices[0].message.content
    except Exception as e:
        return f"查询失败: {e}"

# 创建 Gradio 界面
with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# 实战: 天气查询助手 (Function Calling) ")
    with gr.Row():
        with gr.Column(scale=2):
            query = gr.Textbox(label="请输入", lines=6)
        with gr.Column(scale=1):
            model_input = gr.Radio(["openai", "ZhipuAI", "Bailian"], label="模型", value="openai")
            temperature = gr.Slider(minimum=0.0, maximum=1.0, label="temperature", value=0.8, step=0.1)
            submit_button = gr.Button("提交", size="lg")
    with gr.Row():
        text_output = gr.Textbox(label="模型回复", lines=3)
    # 定义按钮点击事件的回调函数
    submit_button.click(
        fn=chatbot_interface,
        inputs=[query, model_input, temperature],
        outputs=[text_output]
    )
    # 添加 Examples 组件
    examples = [
        "今天北京天气怎么样？",
        "今天北京的空气质量如何？",
        "未来几天北京的天气怎么样？"
    ]
    gr.Examples(examples, [query])
demo.launch()
