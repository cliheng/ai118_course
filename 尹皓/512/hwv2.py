
from openai import OpenAI
import json
import os

client = OpenAI(api_key="sk-e1c976043cbf4bf5b66e4fd79efe3f1c", base_url="https://api.deepseek.com")
HISTORY_FILE = "chat_history.json"

# 初始化聊天历史列表
chat_history = []

# 加载历史记录
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# 保存历史记录
def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# 显示历史记录
def show_history():
    if not chat_history:
        print("暂无历史记录")
        return
    print("\n=== 历史对话记录 ===")
    for i, msg in enumerate(chat_history, 1):
        role = "用户" if msg["role"] == "user" else "AI助手"
        print(f"{i}. [{role}] {msg['content']}")
    print("==================\n")

# 加载已有历史记录
chat_history = load_history()

while True:
    user_input = input("请输入您的问题/命令(输入'history'查看历史,'clear'清空历史,'拜拜'退出): ").strip()
    
    if user_input.lower() == "history":
        show_history()
        continue
    elif user_input.lower() == "clear":
        chat_history = []
        save_history(chat_history)
        print("历史记录已清空")
        continue
    elif user_input.lower() in ["拜拜", "bye"]:
        save_history(chat_history)
        break
    
    # 将用户输入添加到历史记录
    chat_history.append({"role": "user", "content": user_input})
    
    # 构建消息列表，包含系统消息和完整历史记录
    messages = [
        {"role": "system", "content": "你是一个聊天机器人,致力于聊天"}
    ] + chat_history
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )

    # 获取AI回复并添加到历史记录
    ai_response = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": ai_response})
    
    print(ai_response)
    save_history(chat_history)