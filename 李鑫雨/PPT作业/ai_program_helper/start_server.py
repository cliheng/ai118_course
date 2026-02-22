import chromadb
from chromadb.config import Settings

# 配置服务器设置
settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="db",  # 数据持久化目录
    anonymized_telemetry=False
)

# 创建服务器实例
server = chromadb.Server(settings=settings)

# 在8000端口启动服务器
server.run(host='localhost', port=8000)
