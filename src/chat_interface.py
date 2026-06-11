"""Web 对话界面"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

from loguru import logger
import gradio as gr

from src.basic_rag import BasicRAGSystem
from src.config import config


class LegalChatInterface:
    """药典助手对话界面"""

    def __init__(self):
        self.rag_system = BasicRAGSystem()
        self.chat_history = []

    def respond(self, message: str, history: list) -> str:
        """响应函数"""
        if not message or not message.strip():
            return "请输入您的问题"

        # 调用 RAG 系统
        result = self.rag_system.ask(message)

        if result['success']:
            # 添加引用来源
            answer = result['answer']
            if result['source_documents']:
                answer += "\n\n---\n**📚 参考依据**\n"
                for i, doc in enumerate(result['source_documents'][:3], 1):
                    source = doc.metadata.get('source', '未知')
                    preview = doc.page_content[:100].replace('\n', ' ')
                    answer += f"\n[{i}] {preview}... (来源: {source})"

            return answer
        else:
            return f"抱歉，处理您的问题时出错: {result['answer']}"

    def launch(self, share: bool = False, server_port: int = 7860):
        """启动 Web 界面"""
        demo = gr.ChatInterface(
            fn=self.respond,
            title="🤖 智能药典助手 pharmaEase",
            description="""
            ### 基于《中国药典》的智能问答系统

            💡 **可以这样提问：**
            - 白喉抗毒素的制备方法？

            ⚠️ **注意：** 本系统仅供参考。
            """,
            examples=[
                "白喉抗毒素的制备方法？",
            ],
            cache_examples=False,
        )

        logger.info(f"启动 Web 界面: http://localhost:{server_port}")
        demo.launch(share=share, server_port=server_port)


def main():
    """主入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Pharmacopoeia RAG 系统")
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    parser.add_argument("--port", type=int, default=7860, help="服务器端口")
    parser.add_argument("--init-db", action="store_true", help="初始化向量数据库")

    args = parser.parse_args()

    if args.init_db:
        from src.document_loader import DocumentLoader
        from src.vector_store import VectorStoreManager
        from src.config import DATA_DIR

        print("📦 初始化向量数据库...")
        loader = DocumentLoader()
        vs_manager = VectorStoreManager()

        # 删除旧数据库
        vs_manager.delete_collection()

        # 加载所有文档
        if DATA_DIR.exists():
            chunks = loader.load_directory(DATA_DIR)
            vs_manager.create_vector_store(chunks)
            print("✅ 向量数据库初始化完成")
        else:
            print(f"❌ 数据目录不存在: {DATA_DIR}")
        return

    # 启动对话界面
    interface = LegalChatInterface()
    interface.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()