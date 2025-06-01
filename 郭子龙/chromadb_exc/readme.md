实现思路:
1.设计函数parse json，通过传入json文件名,实现json格式数据加载读取,返回[{"keys":"干化词","content":"内容"},...]
2.调用闭源API实现Embedding，测试
3.调用本地开源模型，实现Embedding，测试
4.chroma创建并存储向量记录及关联文本
5.测试问题输入转换为Embedding后通过chroma查询关联文本
    5.1 接收用户输入问题
    5.2 输入问题转换Embedding
    5.3 通过chroma查询问题的Embedding匹配记录5.4 查询结果content，拼接到prompt中。
6.关联文本通过prompt导入智谱大模型,获取结果