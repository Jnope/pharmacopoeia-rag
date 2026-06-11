"""基础 RAG 系统实现"""
from typing import Dict, Any, List

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from src.config import config
from src.vector_store import VectorStoreManager


class BasicRAGSystem:
    """基础 RAG 系统"""

    def __init__(self):
        """初始化 RAG 系统"""
        # 初始化 LLM
        if not config.is_openai_configured:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")

        self.llm = ChatOpenAI(
            model=config.openai_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.openai_api_key,
            base_url=config.openai_api_base
        )

        # 初始化向量存储
        self.vs_manager = VectorStoreManager()
        self.vector_store = self.vs_manager.load_vector_store()

        # 创建检索器
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.retriever_k}
        )

        # 创建提示模板
        self.prompt = self._create_prompt_template()

        # 创建 QA 链
        self.qa_chain = self._create_qa_chain()

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """创建提示模板"""
        return ChatPromptTemplate.from_template("""
你是一位专业的药典顾问，基于以下药典内容回答用户问题。

相关药典内容：
{context}

用户问题：{question}

要求：
1. 严格基于提供的药典内容回答
2. 如果药典中没有明确说明，请说明"药典内容未明确规定"
3. 如果适用，请引用具体药典内容

回答：
""")

    def _create_qa_chain(self) -> RetrievalQA:
        """创建 QA 链"""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True,
            verbose=False
        )

    def ask(self, question: str) -> Dict[str, Any]:
        """提问并获取回答"""
        logger.debug(f"用户问题: {question}")

        try:
            result = self.qa_chain.invoke({"query": question})

            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": result["source_documents"],
                "success": True
            }

            logger.success(f"回答生成成功，长度: {len(result['result'])}")
            return response

        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return {
                "question": question,
                "answer": f"抱歉，生成回答时出错: {str(e)}",
                "source_documents": [],
                "success": False
            }

    def ask_with_context(self, question: str) -> None:
        """提问并打印详细信息"""
        result = self.ask(question)

        print("\n" + "="*60)
        print(f"📝 问题: {result['question']}")
        print(f"🤖 回答: {result['answer']}")

        if result['source_documents']:
            print(f"\n📚 参考来源 ({len(result['source_documents'])} 个):")
            for i, doc in enumerate(result['source_documents'], 1):
                preview = doc.page_content[:150].replace('\n', ' ')
                print(f"  [{i}] {preview}...")
        print("="*60)


# 使用示例
if __name__ == "__main__":
    # 确保向量数据库已创建
    from src.document_loader import DocumentLoader
    from src.vector_store import VectorStoreManager

    # 检查是否需要初始化向量库
    vs_manager = VectorStoreManager()
    if not vs_manager.persist_path.exists():
        print("📦 首次运行，正在初始化向量数据库...")
        loader = DocumentLoader()
        chunks = loader.load_text_file(config.DATA_DIR / "labor_law.txt")
        vs_manager.create_vector_store(chunks)

    # 启动 RAG 系统
    rag = BasicRAGSystem()

    # 测试问题
    questions = [
        "白喉抗毒素的制备方法？",
    ]

    for q in questions:
        rag.ask_with_context(q)