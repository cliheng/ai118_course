# 从 langchain_chroma 模块导入 Chroma 类，用于操作 Chroma 向量数据库
from langchain_chroma import Chroma
# 从 langchain_openai 模块导入 OpenAIEmbeddings 类，用于生成 OpenAI 嵌入向量
from langchain_openai import OpenAIEmbeddings
# 从 langchain_community.document_loaders 模块导入 PDFMinerLoader 类，用于加载 PDF 文件
from langchain_community.document_loaders import PDFMinerLoader
# 导入 os 模块，用于与操作系统进行交互，如文件路径操作
import os 
# 从 langchain_text_splitters 模块导入 RecursiveCharacterTextSplitter 类，用于将文档分割成小块
from langchain_text_splitters import RecursiveCharacterTextSplitter

class MyChroma(Chroma):
    """
    自定义的 Chroma 类，继承自 Chroma，用于扩展 Chroma 向量数据库的功能。
    提供了添加单个 PDF 文件和添加文件夹中所有 PDF 文件的方法。
    """
    def add_file(self, filename):
        """
        将单个 PDF 文件添加到 Chroma 向量数据库集合中。

        参数
        ----------
        :param filename: PDF 文件的路径。
        """
        # 使用 PDFMinerLoader 加载指定路径的 PDF 文件
        document = PDFMinerLoader(filename).load()
        # 使用 RecursiveCharacterTextSplitter 将加载的文档分割成小块，每个块大小为 200 字符，块之间重叠 40 字符
        splits = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40).split_documents(document)
        # 将分割后的文档块添加到 Chroma 向量数据库集合中
        self.add_documents(splits)

    @classmethod
    def add_folder(cls, persist_directory, collection_name, folder_path):
        """
        将指定文件夹中的所有 PDF 文件添加到 Chroma 向量数据库集合中。

        参数
        ----------
        :param persist_directory: 向量数据库持久化存储的目录。
        :param collection_name: 向量数据库集合的名称。
        :param folder_path: 包含 PDF 文件的文件夹路径。

        返回
        -------
        :return: MyChroma 类的实例对象。
        """
        # 创建 OpenAIEmbeddings 对象，用于生成文本的嵌入向量
        embedding_function = OpenAIEmbeddings()
        
        # 创建 MyChroma 类的对象
        # MyChroma == cls，这里使用类方法创建实例
        obj = cls(collection_name, embedding_function, persist_directory)
        
        if folder_path:
            # 获取指定文件夹中所有以 .pdf 结尾的文件的完整路径
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
            # 遍历所有 PDF 文件，调用 add_file 方法将文件添加到向量数据库中
            for f in files:
                obj.add_file(f)
                
        return obj

if __name__ == "__main__":
    # 设置 OpenAI API 密钥，这里为空，需要填入实际的密钥
    os.environ['OPENAI_API_KEY']=''
    # 设置 OpenAI API 的基础地址
    os.environ['OPENAI_API_BASE']='https://oa.api2d.net'

    # 调用 add_folder 类方法，将指定文件夹中的 PDF 文件添加到向量数据库中
    # 扩展后才能实现
    chroma = MyChroma.add_folder('./files/rag', 'rag_collection','files/docs')
    
    # 从向量数据库中获取所有文档
    documents = chroma.get()
    # 获取文档的数量
    n_documents = len(documents['ids'])
    # 遍历所有文档，打印文档的基本信息
    for i in range(n_documents):
        # 处理文档内容，去除换行符和多余的空格
        text = documents['documents'][i].replace('\\n', '').replace('  ', '')
        # 打印文档的编号、ID 前 10 个字符以及内容的前 20 个字符和后 20 个字符
        print(f"Document {i}: {documents['ids'][i][:10]}... 内容: {text[:20]}...{text[-20:]}")
