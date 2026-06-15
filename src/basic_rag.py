"""基础 RAG 系统实现 - 基于 LangGraph Agent"""
import sqlite3
from typing import Dict, Any, List, Optional

from langchain.agents import create_agent
from langchain_core.tools import create_retriever_tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver
from loguru import logger

from langchain_openai import ChatOpenAI

from src.config import config
from src.vector_store import VectorStoreManager

SYSTEM_PROMPT = """你是一位专业的药典顾问，请使用检索工具查询药典内容来回答用户问题。

要求：
1. 必须先使用检索工具查询相关药典内容，再基于检索结果回答
2. 严格基于检索到的药典内容回答
3. 如果药典中没有明确说明，请说明"药典内容未明确规定"
4. 如果适用，请引用具体药典内容"""


class BasicRAGSystem:
    """基础 RAG 系统 - 基于 Agent，支持 SQLite 会话持久化"""

    def __init__(self):
        """初始化 RAG 系统"""
        if not config.is_openai_configured:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")

        self.llm = ChatOpenAI(
            model=config.openai_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.openai_api_key,
            base_url=config.openai_api_base
        )

        self.vs_manager = VectorStoreManager()
        self.vector_store = self.vs_manager.load_vector_store()

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.retriever_k}
        )

        self.retriever_tool = self._create_retriever_tool()

        self._sqlite_conn = sqlite3.connect(str(config.checkpoint_db_path), check_same_thread=False)
        self._checkpointer = SqliteSaver(self._sqlite_conn)
        self.agent = self._create_agent()

    def _create_retriever_tool(self):
        """创建检索工具"""
        return create_retriever_tool(
            retriever=self.retriever,
            name="pharmacopoeia_search",
            description="搜索中国药典相关内容。当用户询问药典相关的任何问题时，使用此工具检索相关内容。",
            response_format="content_and_artifact",
        )

    def _create_agent(self):
        """创建智能体"""
        return create_agent(
            model=self.llm,
            tools=[self.retriever_tool],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self._checkpointer,
        )

    def ask(self, question: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """提问并获取回答

        Args:
            question: 用户问题
            thread_id: 会话ID，相同 thread_id 下保留历史对话上下文。
                       为 None 时不保留历史（无状态调用）。
        """
        logger.debug(f"用户问题: {question}, 会话: {thread_id}")

        try:
            config_kwargs = {}
            if thread_id:
                config_kwargs["configurable"] = {"thread_id": thread_id}

            result = self.agent.invoke(
                {"messages": [HumanMessage(content=question)]},
                config=config_kwargs or None,
            )

            answer = ""
            source_documents: List[Document] = []

            for msg in result["messages"]:
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                elif isinstance(msg, ToolMessage) and msg.artifact:
                    source_documents = msg.artifact

            if not answer:
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage) and msg.content:
                        answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                        break

            response = {
                "question": question,
                "answer": answer,
                "source_documents": source_documents,
                "success": True
            }

            logger.success(f"回答生成成功，长度: {len(answer)}")
            return response

        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return {
                "question": question,
                "answer": f"抱歉，生成回答时出错: {str(e)}",
                "source_documents": [],
                "success": False
            }

    def get_session_history(self, thread_id: str) -> List[Dict[str, str]]:
        """获取指定会话的历史消息摘要"""
        try:
            state = self.agent.get_state(
                config={"configurable": {"thread_id": thread_id}}
            )
            history = []
            for msg in state.values.get("messages", []):
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    history.append({"role": "assistant", "content": msg.content})
            return history
        except Exception as e:
            logger.warning(f"获取会话历史失败: {e}")
            return []

    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话及其最新信息

        Returns:
            按 time倒序 的会话列表，每项含 thread_id, message_count, first_message
        """
        try:
            all_tuples = list(self._checkpointer.list(None, limit=1000))
            seen = set()
            sessions = []
            for t in all_tuples:
                thread_id = t.config["configurable"]["thread_id"]
                if thread_id in seen:
                    continue
                seen.add(thread_id)
                history = self.get_session_history(thread_id)
                first_user_msg = ""
                for h in history:
                    if h["role"] == "user":
                        first_user_msg = h["content"][:50]
                        break
                sessions.append({
                    "thread_id": thread_id,
                    "message_count": len(history),
                    "first_message": first_user_msg,
                })
            return sessions
        except Exception as e:
            logger.warning(f"列出会话失败: {e}")
            return []

    def load_session_messages(self, thread_id: str) -> List[Dict[str, str]]:
        """加载指定会话的聊天记录（用于恢复 UI 显示）

        Returns:
            消息列表，每项含 role 和 content
        """
        return self.get_session_history(thread_id)

    def reset_session(self, thread_id: str) -> None:
        """重置指定会话（删除所有历史，保留 thread_id）"""
        try:
            self._checkpointer.delete_thread(thread_id)
            logger.info(f"会话 {thread_id} 已重置")
        except Exception as e:
            logger.warning(f"重置会话失败: {e}")

    def delete_session(self, thread_id: str) -> None:
        """删除指定会话（从 SQLite 中彻底移除）"""
        try:
            self._checkpointer.delete_thread(thread_id)
            logger.info(f"会话 {thread_id} 已删除")
        except Exception as e:
            logger.warning(f"删除会话失败: {e}")

    def close(self):
        """关闭数据库连接"""
        self._sqlite_conn.close()

    def ask_with_context(self, question: str, thread_id: Optional[str] = None) -> None:
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