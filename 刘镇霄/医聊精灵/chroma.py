import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFMinerLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
os.environ["OPENAI_API_KEY"] = 'fk233463-uiUnPTZFWaxcANhuvTvRTxvGrVPAfH7r'
os.environ["OPENAI_API_BASE"] = 'https://oa.api2d.net'
class MyChroma(Chroma):
    embedding_function = None
    def __init__(self, embedding_function=None, persist_directory=None, collection_name=None):
        if embedding_function is None:
            embedding_function = OpenAIEmbeddings()  # 默认使用OpenAI的嵌入模型
        super().__init__(
            embedding_function=embedding_function,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    def add_file(self, filename):
        """
        Add a PDF file to the Chroma collection.

        Parameters
        ---
        :param filename: Path to the PDF file.
        """

        document = PDFMinerLoader(filename).load()
        splits = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40).split_documents(document)
        self.add_documents(splits)

    @classmethod
    def add_folder(cls, persist_directory, collection_name, folder_path=None):
        obj=cls(cls.embedding_function,persist_directory,collection_name)
        if folder_path:
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
            for f in files:
                obj.add_file(f)
        return obj

if __name__ == '__main__':
    from pathlib import Path
    fold_path=Path(r'files/docs')
    obj=MyChroma.add_folder(fold_path)
    print(f"集合中的文档数量: {obj._collection.count()}")  # 添加这行查看文档数量

