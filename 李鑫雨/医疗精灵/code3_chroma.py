# 导入所需的库
from langchain_chroma import Chroma  # Chroma向量数据库
from langchain_openai import OpenAIEmbeddings  # OpenAI的词嵌入模型
from langchain_community.document_loaders import PDFMinerLoader  # 用于加载PDF文件的加载器
import os  # 用于与操作系统交互，如文件路径操作

# 导入用于文本分割的类
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 定义一个自定义的Chroma类，继承自langchain_chroma.Chroma以扩展功能
class MyChroma(Chroma):

    def add_file(self, filename):
        """
        加载单个PDF文件，将其分割成块，并添加到Chroma集合中。

        参数
        ----------
        :param filename: PDF文件的路径。
        """
        # 使用PDFMinerLoader加载PDF文档
        document = PDFMinerLoader(filename).load()
        # 使用RecursiveCharacterTextSplitter将文档分割成更小的块，以便更好地进行嵌入和搜索
        # chunk_size=200: 每个块的大小为200个字符
        # chunk_overlap=40: 每个块之间有40个字符的重叠，以保持上下文的连续性
        splits = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40).split_documents(document)
        # 将分割后的文档块添加到Chroma向量数据库中
        self.add_documents(splits)

    @classmethod
    def add_folder(cls, persist_directory, collection_name, folder_path):
        """
        一个类方法，用于创建一个MyChroma实例，并处理指定文件夹中的所有PDF文件。

        :param persist_directory: 向量数据库持久化存储的目录。
        :param collection_name: 集合的名称。
        :param folder_path: 包含PDF文件的文件夹路径。
        :return: 一个处理完所有PDF文件的MyChroma对象。
        """
        # 初始化OpenAI的嵌入函数，用于将文本转换为向量
        embedding_function = OpenAIEmbeddings()
        # 创建MyChroma类的实例
        obj = cls(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )

        # 如果提供了文件夹路径
        if folder_path:
            # 获取文件夹中所有以.pdf结尾的文件列表
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
            # 遍历所有PDF文件
            for f in files:
                # 调用add_file方法将每个文件添加到数据库中
                obj.add_file(f)
        # 返回创建并填充好数据的MyChroma对象
        return obj


# 当该脚本作为主程序运行时执行以下代码
if __name__ == "__main__":
    # 从环境变量中获取OpenAI的API密钥和基础URL
    # 注意：这里只是引用了环境变量，需要确保在运行前已经设置了'OPENAI_API_KEY'和'OPENAI_API_BASE'
    os.environ['OPENAI_API_KEY']="fk233469-soioFxSD2BLnroQ2A0G1zbhAvj0YrGrE"
    os.environ['OPENAI_API_BASE']="https://oa.api2d.net"

    # 使用add_folder类方法创建一个Chroma实例，并加载指定文件夹中的文档
    # './files/rag' 是数据库持久化存储的路径
    # 'rag_collection' 是集合的名称
    # 'files/docs' 是存放PDF文档的文件夹
    chroma = MyChroma.add_folder('./files/rag', 'rag_collection', './files/rag')

    # --- 验证数据是否成功存入 ---
    # 从数据库中获取所有文档
    print("正在从向量数据库中检索数据...")
    documents = chroma.get()
    # 获取文档的总数
    n_documents = len(documents['ids'])
    print(f"数据库中共有 {n_documents} 个文档块。")

    # 遍历并打印每个文档的信息
    for i in range(n_documents):
        # 清理文本中的换行符和空格，使其更易读
        text = documents['documents'][i].replace('\n', '').replace(' ', '')
        # 打印文档的索引、ID（前10个字符）以及内容的摘要（前20个和后20个字符）
        print(f"文档 {i}: ID(前10位)={documents['ids'][i][:10]}... 内容摘要: {text[:20]}... ...{text[-20:]}")
