"""
Web 模块配置
"""

from dataclasses import dataclass
import os


@dataclass
class WebConfig:
    """Web 服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    ssl_keyfile: str = ""
    ssl_certfile: str = ""
    db_path: str = "/workspace/data/cache/market_data.db"

    def __post_init__(self):
        """从环境变量加载配置"""
        self.host = os.getenv("WEB_HOST", self.host)
        self.port = int(os.getenv("WEB_PORT", str(self.port)))
        self.debug = os.getenv("WEB_DEBUG", "false").lower() == "true"
        self.ssl_keyfile = os.getenv("SSL_KEYFILE", self.ssl_keyfile)
        self.ssl_certfile = os.getenv("SSL_CERTFILE", self.ssl_certfile)
        self.db_path = os.getenv("CACHE_DB_PATH", self.db_path)

    @classmethod
    def from_env(cls) -> "WebConfig":
        """从环境变量创建配置"""
        return cls()
