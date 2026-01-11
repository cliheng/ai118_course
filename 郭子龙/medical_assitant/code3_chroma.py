from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
# 确保使用正确的PDF加载器
from langchain_community.document_loaders import PDFMinerLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

class MyChroma(Chroma):
    def __init__(self, collection_name, embeddings, persist_directory):
        """
        初始化 MyChroma 类。

        :param collection_name: 集合名称
        :param embeddings: 嵌入模型
        :param persist_directory: 持久化目录
        """
        super().__init__(collection_name=collection_name, embedding_function=embeddings, persist_directory=persist_directory)

    def add_file(self,filename):
        """
        Add a PDF to the Chroma collection
        
        Parameters
        ----------
        :param filename: Path to the PDF file.
        """

        document = PDFMinerLoader(filename).load()
        splits = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        ).split_documents(document)
        self.add_documents(splits)

    @classmethod
    def add_folder(cls,persist_directory,collection_name,folder_path):
        """
        从文件夹中批量添加 PDF 文件到 Chroma 集合。

        :param persist_directory: 持久化目录
        :param collection_name: 集合名称
        :param folder_path: 文件夹路径
        :return: MyChroma 对象
        """
        embeddings = OpenAIEmbeddings()

        # 创建 MyChroma 类的对象
        obj = cls(collection_name, embeddings, persist_directory)

        if folder_path:
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
            for f in files:
                obj.add_file(f)

        return obj



if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = "fk233485-3NGExcniblM1z3RNlpNr6ygajpuDFDjv"
    os.environ["OPENAI_API_BASE"] = "https://oa.api2d.net"



    #扩展原有类功能实现
    # 修改为实际存在的PDF文件夹路径
    chroma = MyChroma.add_folder('./files/rag', 'rag_collection', 'files/docs')

    #打印向量数据库数据
    documents = chroma.get()
    n_documents = len(documents['ids'])
    for i in range(n_documents):
        text = documents['documents'][i].replace('\n','').replace(' ','')
        print(f"Document{i}:{documents['ids'][i]:<.10s}...内容:{text[:20]:<20s}...{text[-20]:<20s}")
        
#创建检索器
    retriever = chroma.as_retriever()



