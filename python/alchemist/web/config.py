"""
Web 模块配置
"""

from dataclasses import dataclass
from pathlib import Path
import os

from utils.config import get_data_cache_path


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
        """从环境变量加载配置"""
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
