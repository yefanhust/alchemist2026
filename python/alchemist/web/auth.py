"""
Web 认证模块：密码验证、Session 管理、IP 封禁
"""

import os
import time
import sqlite3
import secrets
from pathlib import Path

import bcrypt
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired


# 渐进式封禁策略：(失败次数阈值, 封禁秒数)
BAN_TIERS = [
    (5, 30 * 60),       # 5次 → 30分钟
    (10, 2 * 60 * 60),  # 10次 → 2小时
    (20, 24 * 60 * 60), # 20次 → 24小时
]

RATE_LIMIT_WINDOW = 60   # 1分钟窗口
RATE_LIMIT_MAX = 5       # 每分钟最多5次
SESSION_MAX_AGE = 24 * 60 * 60  # session 有效期 24小时
COOKIE_NAME = "alchemist_session"


class IPBanManager:
    """IP 封禁管理器，使用 SQLite 存储"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS login_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ip TEXT NOT NULL,
                    attempt_time REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ip_bans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ip TEXT NOT NULL,
                    banned_until REAL NOT NULL,
                    reason TEXT,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_attempts_ip_time "
                "ON login_attempts(ip, attempt_time)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bans_ip "
                "ON ip_bans(ip, banned_until)"
            )
            conn.commit()
        finally:
            conn.close()

    def is_banned(self, ip: str) -> tuple[bool, int]:
        """检查 IP 是否被封禁，返回 (是否封禁, 剩余秒数)"""
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT banned_until FROM ip_bans "
                "WHERE ip = ? AND banned_until > ? "
                "ORDER BY banned_until DESC LIMIT 1",
                (ip, now),
            ).fetchone()
            if row:
                remaining = int(row[0] - now)
                return True, remaining
            return False, 0
        finally:
            conn.close()

    def is_rate_limited(self, ip: str) -> bool:
        """检查 IP 是否超过频率限制"""
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM login_attempts "
                "WHERE ip = ? AND attempt_time > ?",
                (ip, window_start),
            ).fetchone()
            return row[0] >= RATE_LIMIT_MAX
        finally:
            conn.close()

    def record_failed_attempt(self, ip: str):
        """记录一次失败的登录尝试，必要时触发封禁"""
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO login_attempts (ip, attempt_time) VALUES (?, ?)",
                (ip, now),
            )
            conn.commit()

            # 统计最近24小时内的失败次数
            window_start = now - 24 * 60 * 60
            row = conn.execute(
                "SELECT COUNT(*) FROM login_attempts "
                "WHERE ip = ? AND attempt_time > ?",
                (ip, window_start),
            ).fetchone()
            fail_count = row[0]

            # 按阈值从高到低检查，应用最严格的封禁
            for threshold, duration in reversed(BAN_TIERS):
                if fail_count >= threshold:
                    banned_until = now + duration
                    conn.execute(
                        "INSERT INTO ip_bans (ip, banned_until, reason, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (ip, banned_until, f"Failed {fail_count} times in 24h", now),
                    )
                    conn.commit()
                    break
        finally:
            conn.close()

    def clear_attempts(self, ip: str):
        """登录成功后清除该 IP 的失败记录"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM login_attempts WHERE ip = ?", (ip,))
            conn.execute("DELETE FROM ip_bans WHERE ip = ?", (ip,))
            conn.commit()
        finally:
            conn.close()

    def get_recent_fail_count(self, ip: str) -> int:
        """获取最近24小时内的失败次数"""
        now = time.time()
        window_start = now - 24 * 60 * 60
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM login_attempts "
                "WHERE ip = ? AND attempt_time > ?",
                (ip, window_start),
            ).fetchone()
            return row[0]
        finally:
            conn.close()

    def cleanup_expired(self):
        """清理过期的封禁记录和旧的登录尝试"""
        now = time.time()
        old = now - 7 * 24 * 60 * 60  # 7天前
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM ip_bans WHERE banned_until < ?", (now,))
            conn.execute("DELETE FROM login_attempts WHERE attempt_time < ?", (old,))
            conn.commit()
        finally:
            conn.close()


class AuthManager:
    """认证管理器"""

    def __init__(self, password: str, secret_key: str, ban_manager: IPBanManager):
        self.password_hash = bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt()
        )
        self.serializer = URLSafeTimedSerializer(secret_key)
        self.ban_manager = ban_manager

    def verify_password(self, password: str) -> bool:
        """验证密码"""
        return bcrypt.checkpw(
            password.encode("utf-8"), self.password_hash
        )

    def create_session_token(self) -> str:
        """创建 session token"""
        return self.serializer.dumps({"authenticated": True, "t": time.time()})

    def validate_session_token(self, token: str) -> bool:
        """验证 session token"""
        try:
            self.serializer.loads(token, max_age=SESSION_MAX_AGE)
            return True
        except (BadSignature, SignatureExpired):
            return False
