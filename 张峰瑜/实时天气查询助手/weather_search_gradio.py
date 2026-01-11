import gradio as gr
import json
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

# 导入工具函数和定义
import function_tools

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 加载环境变量
load_dotenv()

# 定义模型配置
MODEL_CONFIG = {
    "openai": {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "base_url": os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
    },
    "ZhipuAI": {
        "api_key": os.environ.get("ZHIPU_API_KEY"),
        "base_url": os.environ.get("ZHIPU_API_BASE"),
        "models": ["glm-4", "glm-4-plus", "glm-3-turbo"]
    },
    "Bailian": {
        # 假设的百炼模型配置
        "api_key": os.environ.get("BAILIAN_API_KEY"),
        "base_url": os.environ.get("BAILIAN_API_BASE"),
        "models": ["qwen-turbo", "qwen-plus"]
    }
}

def chatbot_response(query, platform, model_name, temperature, max_tokens):
    """
    核心对话逻辑函数
    """
    if not query or not query.strip():
        return "错误：输入内容不能为空。"
    if not model_name:
        return f"错误：请为平台 '{platform}' 选择一个模型。"

    config = MODEL_CONFIG.get(platform)
    if not config or not config.get("api_key"):
        return f"错误：未找到平台 '{platform}' 的配置或 API 密钥。"

    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"]
    )

    messages = [
        {"role": "system", "content": "你是一个天气查询助手。请根据用户的输入，调用合适的工具来查询天气信息。"},
        {"role": "user", "content": query}
    ]
    
    tools = function_tools.WEATHER_SEARCH

    try:
        # 第一次调用，让模型决定是否使用工具
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens
        )

        response_message = response.choices[0].message

        # 循环处理工具调用
        while response_message.tool_calls:
            messages.append(response_message)
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_to_call = getattr(function_tools, function_name)
                function_args = json.loads(tool_call.function.arguments)
                
                # 调用工具函数
                function_response = function_to_call(**function_args)
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response, ensure_ascii=False),
                })

            # 再次调用模型，附带工具调用的结果
            second_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            response_message = second_response.choices[0].message

        return response_message.content

    except Exception as e:
        return f"发生错误：{str(e)}"

def update_model_dropdown(platform):
    """
    根据选择的平台更新模型下拉菜单
    """
    models = MODEL_CONFIG.get(platform, {}).get("models", [])
    return gr.Dropdown(choices=models, value=models[0] if models else None)

# --- 创建 Gradio 界面 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 实战：天气查询助手 (Function Calling)")

    with gr.Row():
        with gr.Column(scale=3):
            query = gr.Textbox(label="请输入", lines=5, placeholder="例如：今天北京天气怎么样？")
            submit_button = gr.Button("提交", variant="primary", scale=1)
            text_output = gr.Textbox(label="模型回复", lines=8)
            
            examples = [
                "今天北京天气怎么样？",
                "今天上海的空气质量如何？",
                "未来三天天津的天气怎么样？"
            ]
            gr.Examples(examples=examples, inputs=query, label="示例")

        with gr.Column(scale=1):
            gr.Markdown("### 大模型平台")
            model_platform = gr.Radio(
                choices=list(MODEL_CONFIG.keys()), 
                label="平台选择", 
                value="ZhipuAI"
            )
            
            gr.Markdown("### 模型名称")
            initial_models = MODEL_CONFIG["ZhipuAI"]["models"]
            model_name = gr.Dropdown(
                choices=initial_models, 
                label="选择模型", 
                value=initial_models[0]
            )
            
            temperature = gr.Slider(
                minimum=0.0, maximum=1.0, 
                label="temperature", 
                value=0.8, 
                step=0.1
            )
            max_tokens = gr.Slider(
                minimum=0, maximum=2048, 
                label="max_tokens", 
                value=512, 
                step=64
            )

    # 定义事件回调
    submit_button.click(
        fn=chatbot_response,
        inputs=[query, model_platform, model_name, temperature, max_tokens],
        outputs=[text_output]
    )

    model_platform.change(
        fn=update_model_dropdown,
        inputs=[model_platform],
        outputs=[model_name]
    )

demo.launch()