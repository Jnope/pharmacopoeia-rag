"""文档加载与分割模块 - 语义+大小混合切分"""
import re
from typing import List
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document

from src.config import config
from src.vector_store import VectorStoreManager


class DocumentLoader:
    """文档加载器"""

    def __init__(self):
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

        self.size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            length_function=len,
        )

        vs_manager = VectorStoreManager()
        self.embeddings = vs_manager.embeddings

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a_arr = np.array(a)
        b_arr = np.array(b)
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)

    def _hybrid_split(self, documents: List[Document]) -> List[Document]:
        """混合切分：先按大小切分，再按语义合并相邻块"""
        small_chunks = self.size_splitter.split_documents(documents)
        if not small_chunks:
            return []

        texts = [c.page_content.strip() for c in small_chunks]
        metas = [c.metadata for c in small_chunks]

        logger.info(f"大小切分产生 {len(texts)} 个小块，开始语义合并...")

        all_embeddings = self.embeddings.embed_documents(texts)

        similarity_threshold = 0.5

        merged_chunks = []
        current_text = texts[0]
        current_meta = metas[0]

        for i in range(1, len(texts)):
            sim = self._cosine_similarity(all_embeddings[i - 1], all_embeddings[i])

            if sim >= similarity_threshold and len(current_text) + len(texts[i]) <= self.chunk_size * 2:
                sep = "\n" if not current_text.endswith("\n") else ""
                current_text += sep + texts[i]
            else:
                merged_chunks.append(Document(
                    page_content=current_text,
                    metadata={**current_meta, "split_method": "size+semantic"},
                ))
                current_text = texts[i]
                current_meta = metas[i]

        merged_chunks.append(Document(
            page_content=current_text,
            metadata={**current_meta, "split_method": "size+semantic"},
        ))

        logger.info(f"语义合并后剩余 {len(merged_chunks)} 个块")
        return merged_chunks

    def load_text_file(self, file_path: Path) -> List[Document]:
        """加载文本文件"""
        logger.info(f"加载文本文件: {file_path}")
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        return self._split_and_add_metadata(documents, file_path.stem)

    def load_pdf_file(self, file_path: Path) -> List[Document]:
        """加载 PDF 文件"""
        logger.info(f"加载 PDF 文件: {file_path}")
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        return self._split_and_add_metadata(documents, file_path.stem)

    def load_directory(self, directory: Path) -> List[Document]:
        """批量加载目录中的所有文档"""
        all_chunks = []

        for file_path in directory.iterdir():
            if file_path.suffix == '.txt':
                chunks = self.load_text_file(file_path)
            elif file_path.suffix == '.pdf':
                chunks = self.load_pdf_file(file_path)
            else:
                logger.warning(f"不支持的文件类型: {file_path.suffix}")
                continue

            all_chunks.extend(chunks)

        logger.info(f"共加载 {len(all_chunks)} 个文本块")
        return all_chunks

    def _split_and_add_metadata(self, documents: List[Document], source: str) -> List[Document]:
        """分割文档并添加元数据"""
        chunks = self._hybrid_split(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = source
            chunk.metadata["chunk_id"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        logger.info(f"文档 {source} 最终分割为 {len(chunks)} 个块")
        return chunks


if __name__ == "__main__":
    loader = DocumentLoader()

    if (config.DATA_DIR / "2020版中国药典（三部）全本.pdf").exists():
        chunks = loader.load_pdf_file(config.DATA_DIR / "2020版中国药典（三部）全本.pdf")
        print(f"✅ 加载完成: {len(chunks)} 个文本块")
        print(f"示例: {chunks[0].page_content[:100]}...")
