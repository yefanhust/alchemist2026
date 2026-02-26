"""
Web 模块配置
"""

from dataclasses import dataclass
from pathlib import Path
import os

from utils.config import get_data_cache_path


def _load_dotenv():
    """
    主动加载 docker/.env 文件，避免依赖 uvicorn 启动方式。

    当通过裸命令 `nohup uvicorn ...` 启动时，.env 不会被自动 source，
    导致 WEB_AUTH_PASSWORD 等变量缺失。此函数兜底加载，且不覆盖已有环境变量。
    """
    try:
        from dotenv import load_dotenv
        candidates = [
            Path("/workspace/docker/.env"),
            Path(__file__).resolve().parents[4] / "docker" / ".env",
        ]
        for path in candidates:
            if path.is_file():
                load_dotenv(dotenv_path=path, override=False)
                break
    except ImportError:
        pass


@dataclass
class WebConfig:
    """Web 服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    ssl_keyfile: str = ""
    ssl_certfile: str = ""
    db_path: str = ""
    auth_password: str = ""
    secret_key: str = ""
    auth_db_path: str = ""

    def __post_init__(self):
        """从环境变量加载配置（先确保 .env 已加载）"""
        _load_dotenv()
        self.host = os.getenv("WEB_HOST", self.host)
        self.port = int(os.getenv("WEB_PORT", str(self.port)))
        self.debug = os.getenv("WEB_DEBUG", "false").lower() == "true"
        self.ssl_keyfile = os.getenv("SSL_KEYFILE", self.ssl_keyfile)
        self.ssl_certfile = os.getenv("SSL_CERTFILE", self.ssl_certfile)
        self.db_path = os.getenv("CACHE_DB_PATH", "") or get_data_cache_path()
        self.auth_password = os.getenv("WEB_AUTH_PASSWORD", "")
        self.secret_key = os.getenv("WEB_SECRET_KEY", "")
        if not self.auth_db_path:
            cache_dir = str(Path(self.db_path).parent) if self.db_path else "."
            self.auth_db_path = os.path.join(cache_dir, "auth.db")

    @classmethod
    def from_env(cls) -> "WebConfig":
        """从环境变量创建配置"""
        return cls()
