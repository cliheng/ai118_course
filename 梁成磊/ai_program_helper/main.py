"""
项目的主文件，用于实现gradio页面展现
"""

import gradio as gr
import shutil
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import os
from zhipuai import ZhipuAI
import json
from embedding import cloud_embedding
import chromadb
from file_to_chroma import savefile_chroma
from RAG import search


# 处理好的干化词列表
words = []
# 处理好的具体内容列表
detail = []
# 上传文件的路径
file_upload_path = ""
# 将.env文件中的api_key临时存入环境变量
load_dotenv(find_dotenv())
# 创建chroma数据库对象
client1 = chromadb.HttpClient(host="localhost")
# 定义文件保存路径
UPLOAD_DIR = Path("data_path")
# 确保文件保存路径存在
UPLOAD_DIR.mkdir(exist_ok=True)


# 调用大模型进行对话
def llm_chat(message, history):
    """
    调用智谱大模型API实现对话功能
    Args:
        message (str): 用户输入的消息
        history (list): 对话历史记录
    Returns:
        str: 模型生成的回复内容
    """
    client = ZhipuAI(api_key=os.environ["zhipu_key"])  # 请填写您自己的APIKey
    query = cloud_embedding([message],"text-embedding-v3")
    msg = search(query, "combat_collection")
    messages = []
    if msg:
        messages.append(
            {
                "role": "system",
                "content": "你是一个精通python编程开发的专家，可以根据用户的问题给出专业的回答。回答内容请参考补充资料。\n\n补充资料：{msg}",
            }
        )
    if history:
        messages.append({"role": "user", "content": message})
        messages = messages + history
    else:
        messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model="glm-4-flash-250414",  # 请填写您要调用的模型名称
        messages=messages,
    )
    return response.choices[0].message.content


# 对上传的文件数据进行处理
def process_file(file_path):
    """
    对上传的文件中的数据进行处理
    Args:
        file_path (str): 上传的文件路径
    Returns:
        tuple: 包含干化词和文档内容的列表
    """
    # 读取文件
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        dry_words = []
        doc_content = []
        for item in data:
            text = item["k_qa_content"]
            key, value = text.split("#\n")
            dry_words.append(key)
            doc_content.append(value)

    return dry_words, doc_content


# 上传文件的操作
def upload_file(file_obj):
    """
    处理文件上传
    Args:
        file_obj: 上传的文件对象
    Returns:
        str: 文件路径或提示信息
    """
    try:
        if file_obj is not None:
            # 构造目标文件路径
            file_name = Path(file_obj.name).name  # 获取文件名
            save_path = UPLOAD_DIR / file_name
            # 复制文件到目标路径
            shutil.copy2(file_obj.name, save_path)
            # 保存之后进行文件处理
            global words, detail, file_upload_path
            words, detail = process_file(str(save_path.absolute()))
            file_upload_path = str(save_path.absolute())
            # 调用embedding模型将数据转化为向量
            embedding_list = cloud_embedding(words, "text-embedding-v3")
            dict_detail = []
            for i in detail:
                dict_detail.append({"content": i})
            savefile_chroma(
                embedding_list, words, dict_detail, "combat_collection"
            )
            # 返回保存后的文件路径
            return str(save_path.absolute())
        return "未选择文件"
    except Exception as e:
        return f"上传出错：{str(e)}"


# 创建Gradio界面
with gr.Blocks(fill_height=True) as demo:
    with gr.Row(scale=2):
        with gr.Column(scale=2):
            gr.Markdown("# 智能学习助教")
            gr.ChatInterface(
                fn=llm_chat,
                type="messages",
                examples=["介绍一下类和对象", "解释一下面向对象"],
                run_examples_on_click=False,
            )
            gr.ClearButton(value="Clear")
        with gr.Column(scale=1):
            gr.Markdown("## 文件上传")
            file_upload = gr.File(
                label="文件上传",
                file_count="single",
                file_types=[".json"],
                type="filepath",  
            )
            file_path_text = gr.Textbox(label="保存路径", interactive=False)
            file_upload.change(  # 使用 change 事件替代 upload
                fn=upload_file,
                inputs=[file_upload],  # 明确指定输入组件
                outputs=[file_path_text],  # 明确指定输出组件
            )


if __name__ == "__main__":
    demo.launch()
