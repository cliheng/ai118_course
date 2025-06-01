from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())
print(os.getenv("api_key"))  # 输出环境变量中的api_key