import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class MyChroma(Chroma):
    """Chroma 向量数据库的自定义封装类

    继承自 Chroma 类，扩展了文件夹和文件的批量导入功能
    用于构建和管理文档的向量存储
    """

    @classmethod
    def add_folder(cls, persist_directory, collection_name, folder_path):
        """将指定文件夹中的所有 PDF 文件添加到向量数据库

        参数:
            persist_directory (str): 向量数据库持久化存储的目录
            collection_name (str): 集合名称
            folder_path (str): PDF文件所在的文件夹路径

        返回:
            MyChroma: 初始化好的向量数据库实例
        """
        # 创建 OpenAI 的词嵌入函数实例
        embedding_function = OpenAIEmbeddings()
        # 初始化 Chroma 实例
        obj = cls(collection_name, embedding_function, persist_directory)

        if folder_path:
            # 获取文件夹中所有 PDF 文件的完整路径
            files = [
                # 智能地拼接文件路径
                os.path.join(folder_path, f)
                # os.listdir() 返回文件夹中的所有文件和文件夹名称
                for f in os.listdir(folder_path)
                # 判断结尾是否是以 .pdf 结束
                if f.endswith(".pdf")
            ]
            # 遍历处理每个 PDF 文件
            for f in files:
                obj.add_file(f)
        return obj

    def add_file(self, filename):
        """将单个 PDF 文件添加到向量数据库

        处理流程：
        1. 使用 PDFMinerLoader 加载 PDF 文件
        2. 使用 RecursiveCharacterTextSplitter 将文档切分成小块
        3. 将切分后的文档块添加到向量数据库

        参数:
            filename (str): PDF文件的路径
        """
        # 加载 PDF 文件
        document = PDFMinerLoader(filename).load()
        # 将文档切分成小块，设置块大小为200字符，重叠40字符
        splits = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=40
        ).split_documents(document)
        # 将文档块添加到向量数据库
        self.add_documents(splits)
