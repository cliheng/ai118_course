# 导入必要库模块
from langchain_chroma import Chroma  # Chroma向量数据库交互模块
from langchain_openai import OpenAIEmbeddings  # OpenAI嵌入模型模块
from langchain_community.document_loaders import PDFMinerLoader  # PDF文档加载器
import os  # 系统路径操作模块
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归字符文本分割器
from dotenv import load_dotenv  # 环境变量加载模块
import time  # 时间处理模块
load_dotenv()

class MyChroma(Chroma):
    """扩展Chroma类实现PDF文档处理功能
    
    属性说明
    ----------
    继承自langchain_chroma.Chroma所有属性
    """
    
    def add_file(self, filename):
        document = PDFMinerLoader(filename).load()
        splits = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 增加分块大小减少总数
            chunk_overlap=40
        ).split_documents(document)
        
        # 分批处理（每批50个文本块）
        batch_size = 50
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            self.add_documents(batch)  # 小批量提交
            time.sleep(1)  # 添加短暂延迟

    @classmethod
    def add_folder(cls, persist_directory, collection_name, folder_path):
        """
        批量处理文件夹中的PDF文档
        
        参数说明
        ----------
        persist_directory : str
            向量数据库持久化存储目录路径
        collection_name : str
            集合名称（不同集合存储不同文档集）
        folder_path : str
            包含PDF文档的文件夹路径
        """
        # 初始化OpenAI文本嵌入模型
        embedding_function = OpenAIEmbeddings(
            model="text-embedding-ada-002",  # 使用GPT-4o嵌入模型
        )

        # 创建MyChroma实例
        obj = cls(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )

        if folder_path:
            # 获取文件夹中所有PDF文件路径
            files = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path) 
                if f.endswith('.pdf')
            ]
            # 遍历处理每个PDF文件
            for f in files:
                obj.add_file(f)
        return obj

if __name__ == "__main__":
    # 配置OpenAI API凭证（生产环境应使用更安全的方式存储）

    # 初始化向量数据库（参数说明）
    # persist_directory: 持久化存储路径
    # collection_name: 集合名称
    # folder_path: PDF文档存储路径
    chroma = MyChroma.add_folder(
        persist_directory='./0718/files/rag',
        collection_name='rag_collection',
        folder_path='d:\\Trae CN\\7.21\\0718\\files\\docs'
    )

    # 获取并打印向量库元数据
    documents = chroma.get()
    n_documents = len(documents['ids'])
    # 格式化输出文档摘要
    for i in range(n_documents):
        # 清理文档文本格式
        text = f"{documents['documents'][i]}".replace('\n', '').replace(' ', '')
        # 输出文档ID和首尾各20字符内容
        print(f"Document {i}: {documents['ids'][i]:<.10s}... 内容: {text[:20]:<20s}...<{text[-20:]:<20s}>")

    # 创建检索器实例（用于后续相似性搜索）
    retriever = chroma.as_retriever()