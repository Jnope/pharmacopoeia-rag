"""项目配置管理"""
import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

# 加载环境变量
load_dotenv()

# HuggingFace 离线模式，优先使用本地缓存
if os.getenv("HF_HUB_OFFLINE", "1") != "0":
    os.environ["HF_HUB_OFFLINE"] = "1"

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
VECTOR_STORE_DIR = ROOT_DIR / "chroma_db"
MODELS_DIR = ROOT_DIR / "models"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class Config:
    """全局配置"""
    # OpenAI 配置
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_api_base: str = os.getenv("OPENAI_API_BASE", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # Embedding 配置
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    use_openai_embedding: bool = os.getenv("USE_OPENAI_EMBEDDING", "false").lower() == "true"
    hf_token: str = os.getenv("HF_TOKEN", "")

    # RAG 配置
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    retriever_k: int = int(os.getenv("RETRIEVER_K", "4"))

    # LLM 配置
    temperature: float = float(os.getenv("TEMPERATURE", "0.3"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1000"))

    # 数据库配置
    vector_store_path: Path = VECTOR_STORE_DIR
    collection_name: str = os.getenv("COLLECTION_NAME", "pharma_collection")

    # 会话持久化配置
    checkpoint_db_path: Path = ROOT_DIR / "chat_history.db"

    @property
    def is_openai_configured(self) -> bool:
        """检查 OpenAI 是否已配置"""
        return bool(self.openai_api_key) and bool(self.openai_api_base)


config = Config()