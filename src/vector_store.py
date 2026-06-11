"""向量数据库管理模块"""
from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.config import config, ROOT_DIR


class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self):
        self.embeddings = self._init_embeddings()
        self.persist_path = config.vector_store_path
        self.collection_name = config.collection_name

    def _init_embeddings(self):
        """初始化 Embedding 模型"""
        if config.use_openai_embedding and config.is_openai_configured:
            logger.info("使用 OpenAI Embedding 模型")
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=config.openai_api_key,
                base_url=config.openai_api_base or None
            )
        else:
            local_model_path = ROOT_DIR / "models" / "bge-small-zh-v1.5"
            if local_model_path.exists():
                logger.info(f"使用本地 Embedding 模型: {local_model_path}")
                model_name = str(local_model_path)
            else:
                logger.info(f"使用 Embedding 模型: {config.embedding_model}")
                model_name = config.embedding_model
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """创建向量数据库"""
        logger.info(f"创建向量数据库，文档数: {len(documents)}")

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.persist_path),
            collection_name=self.collection_name
        )

        logger.info(f"✅ 向量数据库已创建: {self.persist_path}")
        logger.info(f"📊 存储向量数: {vector_store._collection.count()}")

        return vector_store

    def load_vector_store(self) -> Chroma:
        """加载已存在的向量数据库"""
        logger.info(f"加载向量数据库: {self.persist_path}")

        vector_store = Chroma(
            persist_directory=str(self.persist_path),
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

        count = vector_store._collection.count()
        logger.info(f"✅ 加载已有向量库，包含 {count} 个向量")

        return vector_store

    def similarity_search(
            self,
            vector_store: Chroma,
            query: str,
            k: int = None,
            filter_condition: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """相似度检索"""
        k = k or config.retriever_k

        if filter_condition:
            results = vector_store.similarity_search(query, k=k, filter=filter_condition)
        else:
            results = vector_store.similarity_search(query, k=k)

        logger.debug(f"检索查询: {query[:50]}...，返回 {len(results)} 个结果")
        return results

    def add_documents(self, vector_store: Chroma, documents: List[Document]) -> None:
        """增量添加文档"""
        logger.info(f"增量添加 {len(documents)} 个文档")

        vector_store.add_documents(documents)
        logger.info(f"✅ 添加完成，当前向量数: {vector_store._collection.count()}")

    def delete_collection(self) -> None:
        """删除整个集合"""
        import shutil
        if self.persist_path.exists():
            shutil.rmtree(self.persist_path)
            logger.warning(f"已删除向量数据库: {self.persist_path}")


# 使用示例
if __name__ == "__main__":
    from src.document_loader import DocumentLoader

    # 加载文档
    loader = DocumentLoader()
    chunks = loader.load_text_file(config.DATA_DIR / "2020版中国药典（三部）全本.pdf")

    # 创建向量库
    vs_manager = VectorStoreManager()
    vector_store = vs_manager.create_vector_store(chunks)

    # 测试检索
    query = "白喉抗毒素的制备方法？"
    results = vs_manager.similarity_search(vector_store, query)

    print(f"\n🔍 查询: {query}")
    for i, doc in enumerate(results):
        print(f"结果 {i+1}: {doc.page_content[:100]}...")