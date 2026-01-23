"""管理员认证模块

实现管理员登录认证功能：
- 密码哈希存储（PBKDF2）
- Token 生成与验证
- 登录限速
- 会话管理
"""
import hashlib
import hmac
import secrets
import time
import uuid
from base64 import b64encode, b64decode
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

from .persistence import load_config, save_config


@dataclass
class AdminTokenRecord:
    """管理员 Token 记录"""
    id: str              # uuid4 hex
    token_hash: str      # sha256 hex
    created_at: float    # unix timestamp
    expires_at: float    # unix timestamp
    last_used: float     # unix timestamp


class LoginRateLimiter:
    """登录限速器 - 5次/分钟"""

    def __init__(self, max_attempts: int = 5, window_seconds: int = 60):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts: Dict[str, List[float]] = {}

    def is_allowed(self, client_ip: str) -> bool:
        """检查是否允许登录尝试"""
        now = time.time()

        # 清理过期记录
        if client_ip in self._attempts:
            self._attempts[client_ip] = [
                t for t in self._attempts[client_ip]
                if now - t < self.window_seconds
            ]

        # 检查当前窗口内的尝试次数
        attempts = self._attempts.get(client_ip, [])
        return len(attempts) < self.max_attempts

    def record_attempt(self, client_ip: str):
        """记录登录尝试"""
        now = time.time()
        if client_ip not in self._attempts:
            self._attempts[client_ip] = []
        self._attempts[client_ip].append(now)


# 全局限速器实例
_rate_limiter = LoginRateLimiter()


def pbkdf2_hash(password: str, salt: bytes, iterations: int = 100000) -> str:
    """使用 PBKDF2 对密码进行哈希"""
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations).hex()


def verify_password(password: str, salt_b64: str, expected_hash: str) -> bool:
    """验证密码"""
    try:
        salt = b64decode(salt_b64)
        actual_hash = pbkdf2_hash(password, salt)
        return hmac.compare_digest(actual_hash, expected_hash)
    except Exception:
        return False


def generate_token() -> str:
    """生成随机 Token"""
    return secrets.token_hex(32)  # 64 字符


def hash_token(token: str) -> str:
    """对 Token 进行哈希"""
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def add_token_record(config: Dict[str, Any], token: str, ttl_seconds: int = 7 * 24 * 3600) -> AdminTokenRecord:
    """添加 Token 记录到配置"""
    now = time.time()

    record = AdminTokenRecord(
        id=uuid.uuid4().hex,
        token_hash=hash_token(token),
        created_at=now,
        expires_at=now + ttl_seconds,
        last_used=now
    )

    # 确保 admin_tokens 列表存在
    if "admin_tokens" not in config:
        config["admin_tokens"] = []

    config["admin_tokens"].append(asdict(record))
    return record


def find_valid_token(config: Dict[str, Any], token_hash: str, now: float) -> Optional[AdminTokenRecord]:
    """查找有效的 Token 记录"""
    tokens = config.get("admin_tokens", [])

    for token_data in tokens:
        record = AdminTokenRecord(**token_data)

        # 检查哈希匹配和过期时间
        if (hmac.compare_digest(record.token_hash, token_hash) and
            record.expires_at > now):
            return record

    return None


def update_token_last_used(config: Dict[str, Any], token_hash: str, now: float) -> bool:
    """更新 Token 最后使用时间"""
    tokens = config.get("admin_tokens", [])

    for token_data in tokens:
        if hmac.compare_digest(token_data["token_hash"], token_hash):
            token_data["last_used"] = now
            return True

    return False


def revoke_token(config: Dict[str, Any], token_hash: str) -> bool:
    """撤销单个 Token"""
    tokens = config.get("admin_tokens", [])
    original_count = len(tokens)

    config["admin_tokens"] = [
        token for token in tokens
        if not hmac.compare_digest(token["token_hash"], token_hash)
    ]

    return len(config["admin_tokens"]) < original_count


def revoke_all_tokens(config: Dict[str, Any]):
    """撤销所有 Token"""
    config["admin_tokens"] = []


def prune_expired_tokens(config: Dict[str, Any], now: float) -> int:
    """清理过期 Token，返回清理数量"""
    tokens = config.get("admin_tokens", [])
    original_count = len(tokens)

    config["admin_tokens"] = [
        token for token in tokens
        if token["expires_at"] > now
    ]

    return original_count - len(config["admin_tokens"])


def setup_admin_password(password: str) -> Tuple[str, str]:
    """设置管理员密码，返回 (salt_b64, password_hash)"""
    salt = secrets.token_bytes(32)
    salt_b64 = b64encode(salt).decode('ascii')
    password_hash = pbkdf2_hash(password, salt)
    return salt_b64, password_hash


def is_admin_password_configured() -> bool:
    """检查是否已配置管理员密码"""
    config = load_config()
    return bool(config.get("admin_password_hash") and config.get("admin_salt"))


def get_rate_limiter() -> LoginRateLimiter:
    """获取全局限速器实例"""
    return _rate_limiter