from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PDFMinerLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
class MyChroma(Chroma):
    def add_file(self, filename):
        """
        Add a PDF file to the Chroma collection.

        Parameters
        ----------
        :param filename: Path to the PDF file.
        """
        document = PDFMinerLoader(filename).load()
        splits = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40).split_documents(document)
        self.add_documents(splits)
    @classmethod
    def add_folder(cls, persist_directory, collection_name, folder_path):
        embedding_function = OpenAIEmbeddings()
        # 创建MyChroma类的对象
        obj = cls(collection_name, embedding_function, persist_directory)

        if folder_path:
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
            for f in files:
                obj.add_file(f)
        return obj
if __name__ == "__main__":
    os.environ['OPENAI_API_KEY'] = 'k233465-7O1DxqKcVWmEhcArShuZ4nSmznkIe12H'
    os.environ['OPENAI_API_BASE'] = 'https://oa.api2d.net'

    # 扩展原有类功能实现
    chroma = MyChroma.add_folder('./files/rag', 'rag_collection', 'files/docs')

    # 打印向量数据库数据
    documents = chroma.get()
    n_documents = len(documents['ids'])
    for i in range(n_documents):
        text = documents['documents'][i].replace('\n', '').replace(' ', '')
        print(f"Document {i}: {documents['ids'][i] [:10]}... 内容: {text[:20]}...<{text[-20:]}>")

    # 创建检索器
    retriever = chroma.as_retriever()

    