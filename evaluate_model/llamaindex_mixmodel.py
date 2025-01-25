import chromadb
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    get_response_synthesizer,
    Settings,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore


Settings.embed_model = OllamaEmbedding(model_name="yxl/m3e:latest")
# 创建 Ollama 客户端，两个模型分别用于检索和回答生成
model1_client = Ollama(model="qwen2.5:14b", request_timeout=260.0)  # 检索模型
model2_client = Ollama(model="yaowen2:latest", request_timeout=260.0)  # 回答生成模型

# 连接到 Chroma 数据库
client = chromadb.Client()

# 初始化 Chroma 客户端，指定数据存储路径为当前目录下的 chroma_db 文件夹
db = chromadb.PersistentClient(path="/root/autodl-tmp/evaluate_model/data/chroma_db")

# 获取或创建名为 "quickstart" 的集合，如果该集合不存在，则创建它
chroma_collection = db.get_or_create_collection("quickstart")

# 使用上述集合创建一个 ChromaVectorStore 实例，以便 llama_index 可以与 Chroma 集合进行交互
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 创建一个存储上下文，指定向量存储为刚刚创建的 ChromaVectorStore 实例
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 从向量存储创建索引
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


retriever = VectorIndexRetriever(index=index, llm=model1_client)
response_synthesizer = get_response_synthesizer(llm=model2_client)
query_engine = RetrieverQueryEngine(
    retriever=retriever, response_synthesizer=response_synthesizer
)

while True:
    question = input("请输入您的问题（输入q退出）：")
    if question.lower() == "q":
        break
    response = query_engine.query(question)
    print(response)
