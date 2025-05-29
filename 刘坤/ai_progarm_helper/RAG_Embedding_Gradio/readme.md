# RAG聊天系统

## 项目介绍

这是一个简单的RAG（检索增强生成）聊天系统，可以根据知识库回答问题。

## 功能说明

1. 用户输入问题
2. 系统在知识库中搜索相关内容
3. 结合搜索结果生成回答

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动数据库
```bash
python chroma.py
```

### 运行程序
```bash
python main.py
```

### 访问系统
打开浏览器访问：http://localhost:7860

## 技术组件

- **向量嵌入**：BAAI/bge-large-zh-v1.5
- **文本生成**：gemini-2.5-pro-exp-03-25
- **向量数据库**：ChromaDB
- **用户界面**：Gradio

## 文件说明

- `main.py` - 主程序
- `chroma.py` - 数据库初始化
- `data_source.json` - 知识库数据
- `requirements.txt` - 依赖包
- `start.py` - 启动脚本
- `test_api.py` - API测试

## 注意事项

1. 确保ChromaDB服务运行在localhost:8000
2. 检查API密钥配置
3. 首次运行需要初始化数据库

## 故障排除

### 常见问题

1. **ChromaDB连接失败**
   - 检查ChromaDB服务是否启动
   - 确认端口8000未被占用

2. **API调用失败**
   - 检查网络连接
   - 验证API密钥是否有效

3. **图片处理失败**
   - 确认图片格式支持
   - 检查图片文件是否损坏

    