"""Web 对话界面"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import uuid
from loguru import logger
import gradio as gr

from src.basic_rag import BasicRAGSystem
from src.config import config


class LegalChatInterface:
    """药典助手对话界面 - 支持多会话"""

    def __init__(self):
        self.rag_system = BasicRAGSystem()
        self._session_id = str(uuid.uuid4())

    def _new_session(self) -> str:
        self._session_id = str(uuid.uuid4())
        logger.info(f"新会话创建: {self._session_id}")
        return self._session_id

    def _build_session_choices(self):
        sessions = self.rag_system.list_sessions()
        choices = []
        for s in sessions:
            tid = s["thread_id"]
            label = f"{tid[:8]}... | {s['first_message']}..." if s["first_message"] else f"{tid[:8]}... | (空)"
            choices.append((label, tid))
        return choices

    def _get_session_choices(self):
        choices = self._build_session_choices()
        current = self._session_id
        return gr.update(choices=choices, value=current), f"当前会话: {self._session_id}"

    def respond(self, message: str, history: list) -> str:
        if not message or not message.strip():
            return "请输入您的问题"

        result = self.rag_system.ask(message, thread_id=self._session_id)

        if result['success']:
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

    def _on_new_session(self):
        self._new_session()
        chatbot_value = []
        dropdown_update, display_text = self._get_session_choices()
        return chatbot_value, dropdown_update, display_text

    def _on_reset_session(self):
        self.rag_system.reset_session(self._session_id)
        dropdown_update, display_text = self._get_session_choices()
        return [], dropdown_update, display_text

    def _on_delete_session(self, selected_thread_id):
        if not selected_thread_id:
            dropdown_update, display_text = self._get_session_choices()
            return [], dropdown_update, display_text
        self.rag_system.delete_session(selected_thread_id)
        if selected_thread_id == self._session_id:
            self._new_session()
        dropdown_update, display_text = self._get_session_choices()
        return [], dropdown_update, display_text

    def _on_switch_session(self, selected_thread_id):
        if selected_thread_id:
            self._session_id = selected_thread_id
        messages = self.rag_system.load_session_messages(self._session_id)
        chatbot_value = []
        for m in messages:
            chatbot_value.append({"role": m["role"], "content": m["content"]})
        _, display_text = self._get_session_choices()
        return chatbot_value, display_text

    def _on_refresh_sessions(self):
        dropdown_update, display_text = self._get_session_choices()
        return dropdown_update, display_text

    def launch(self, share: bool = False, server_port: int = 7860):
        with gr.Blocks(title="智能药典助手 pharmaEase") as demo:
            gr.Markdown("""
            # 🤖 智能药典助手 pharmaEase
            ### 基于《中国药典》的智能问答系统

            💡 **可以这样提问：**
            - 白喉抗毒素的制备方法？

            ⚠️ **注意：** 本系统仅供参考。对话历史自动保存至本地 SQLite，刷新页面不会丢失。
            """)

            with gr.Row():
                initial_choices = self._build_session_choices()
                initial_value = self._session_id
                session_dropdown = gr.Dropdown(
                    choices=initial_choices,
                    value=initial_value,
                    label="会话列表",
                    scale=3,
                    allow_custom_value=True,
                    interactive=True,
                )

            with gr.Row():
                refresh_btn = gr.Button("刷新", variant="secondary", scale=1)
                reset_btn = gr.Button("重置会话", variant="secondary", scale=1)
                new_btn = gr.Button("新建会话", variant="secondary", scale=1)
                delete_btn = gr.Button("删除选中会话", variant="stop", scale=1)

            session_display = gr.Textbox(
                value=f"当前会话: {self._session_id}",
                label="会话信息",
                interactive=False,
            )

            chatbot = gr.ChatInterface(
                fn=self.respond,
                cache_examples=False,
            )

            new_btn.click(
                fn=self._on_new_session,
                outputs=[chatbot.chatbot, session_dropdown, session_display],
            )
            delete_btn.click(
                fn=self._on_delete_session,
                inputs=[session_dropdown],
                outputs=[chatbot.chatbot, session_dropdown, session_display],
            )
            session_dropdown.change(
                fn=self._on_switch_session,
                inputs=[session_dropdown],
                outputs=[chatbot.chatbot, session_display],
            )
            refresh_btn.click(
                fn=self._on_refresh_sessions,
                outputs=[session_dropdown, session_display],
            )
            reset_btn.click(
                fn=self._on_reset_session,
                outputs=[session_dropdown, session_display],
            )

        self._get_session_choices()
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