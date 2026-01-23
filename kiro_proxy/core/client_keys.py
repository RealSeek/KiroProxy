"""Client Key 管理模块

管理 API 访问密钥，支持：
- 多 Key 管理（稳定 UUID 标识）
- 渐进式安全（无 Key 时开放，有 Key 时强制验证）
- 请求统计（使用次数、最后使用时间）
"""
import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from threading import Lock

from .persistence import CONFIG_DIR, ensure_config_dir


# Client Keys 配置文件
CLIENT_KEYS_FILE = CONFIG_DIR / "client_tokens.json"

# Key 前缀（默认生成时使用）
KEY_PREFIX = "sk-"


@dataclass
class ClientKey:
    """Client Key 数据模型"""
    id: str                              # UUID 稳定标识
    name: str                            # 显示名称
    key_hash: str                        # SHA-256 哈希（不存明文）
    enabled: bool = True                 # 是否启用
    created_at: float = field(default_factory=time.time)  # 创建时间
    last_used_at: Optional[float] = None # 最后使用时间
    usage_count: int = 0                 # 请求次数

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于持久化）"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClientKey":
        """从字典创建"""
        return cls(
            id=data["id"],
            name=data["name"],
            key_hash=data["key_hash"],
            enabled=data.get("enabled", True),
            created_at=data.get("created_at", time.time()),
            last_used_at=data.get("last_used_at"),
            usage_count=data.get("usage_count", 0)
        )

    def to_display_dict(self) -> Dict[str, Any]:
        """转换为显示字典（用于 API 响应，不含敏感信息）"""
        return {
            "id": self.id,
            "name": self.name,
            "prefix": self.key_hash[:12] + "...",  # 显示哈希前缀
            "enabled": self.enabled,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "usage_count": self.usage_count
        }


class ClientKeyManager:
    """Client Key 管理器"""

    def __init__(self):
        self._keys: List[ClientKey] = []
        self._lock = Lock()
        self._dirty = False  # 标记是否有未保存的更改
        self._load()

    def _load(self):
        """从文件加载 Keys"""
        try:
            if CLIENT_KEYS_FILE.exists():
                with open(CLIENT_KEYS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    keys_data = data.get("keys", [])
                    self._keys = [ClientKey.from_dict(k) for k in keys_data]
                    print(f"[ClientKeys] 加载 {len(self._keys)} 个 Client Key")
        except Exception as e:
            print(f"[ClientKeys] 加载失败: {e}")
            self._keys = []

    def _save(self) -> bool:
        """保存 Keys 到文件（原子写入）"""
        try:
            ensure_config_dir()
            data = {
                "keys": [k.to_dict() for k in self._keys],
                "version": 1
            }
            # 原子写入：先写临时文件，再重命名
            temp_file = CLIENT_KEYS_FILE.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_file.replace(CLIENT_KEYS_FILE)
            self._dirty = False
            return True
        except Exception as e:
            print(f"[ClientKeys] 保存失败: {e}")
            return False

    def _hash_key(self, key: str) -> str:
        """计算 Key 的 SHA-256 哈希"""
        return hashlib.sha256(key.encode()).hexdigest()

    def _generate_key(self) -> str:
        """生成新的 API Key"""
        # 生成 32 字节随机数，转为 hex
        random_part = secrets.token_hex(24)
        return f"{KEY_PREFIX}{random_part}"

    def create_key(self, name: str, custom_key: str = None) -> Tuple[str, ClientKey]:
        """
        创建新 Key

        Args:
            name: Key 名称
            custom_key: 自定义 Key（可选，为空则自动生成）

        Returns:
            (明文 key, ClientKey 对象)
            注意：明文 key 仅返回一次，之后无法查看
        """
        with self._lock:
            # 使用自定义 Key 或生成新 Key
            if custom_key and custom_key.strip():
                plain_key = custom_key.strip()
                # 检查是否已存在相同的 Key
                key_hash = self._hash_key(plain_key)
                for existing_key in self._keys:
                    if existing_key.key_hash == key_hash:
                        raise ValueError("该 Key 已存在")
            else:
                plain_key = self._generate_key()

            key_hash = self._hash_key(plain_key)

            # 生成 UUID
            key_id = secrets.token_hex(8)

            # 创建 ClientKey 对象
            client_key = ClientKey(
                id=key_id,
                name=name,
                key_hash=key_hash
            )

            self._keys.append(client_key)
            self._save()

            print(f"[ClientKeys] 创建 Key: {name} (id={key_id})")
            return plain_key, client_key

    def validate_key(self, key: str) -> Optional[ClientKey]:
        """
        验证 Key 是否有效

        Args:
            key: 明文 API Key

        Returns:
            验证成功返回 ClientKey 对象，失败返回 None
        """
        if not key:
            return None

        key_hash = self._hash_key(key)

        with self._lock:
            for client_key in self._keys:
                if client_key.key_hash == key_hash and client_key.enabled:
                    # 更新统计
                    client_key.usage_count += 1
                    client_key.last_used_at = time.time()
                    self._dirty = True
                    return client_key

        return None

    def flush_stats(self):
        """将统计数据写入磁盘（定期调用）"""
        with self._lock:
            if self._dirty:
                self._save()

    def has_enabled_keys(self) -> bool:
        """是否有启用的 Key（用于渐进式安全判断）"""
        with self._lock:
            return any(k.enabled for k in self._keys)

    def get_all_keys(self) -> List[Dict[str, Any]]:
        """获取所有 Key（用于 API 响应）"""
        with self._lock:
            return [k.to_display_dict() for k in self._keys]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total = len(self._keys)
            active = sum(1 for k in self._keys if k.enabled)
            total_requests = sum(k.usage_count for k in self._keys)
            return {
                "total_keys": total,
                "active_keys": active,
                "total_requests": total_requests
            }

    def toggle_key(self, key_id: str) -> Optional[bool]:
        """
        切换 Key 启用/禁用状态

        Returns:
            新状态，如果 Key 不存在返回 None
        """
        with self._lock:
            for client_key in self._keys:
                if client_key.id == key_id:
                    client_key.enabled = not client_key.enabled
                    self._save()
                    print(f"[ClientKeys] 切换 Key {key_id}: enabled={client_key.enabled}")
                    return client_key.enabled
        return None

    def delete_key(self, key_id: str) -> bool:
        """删除 Key"""
        with self._lock:
            for i, client_key in enumerate(self._keys):
                if client_key.id == key_id:
                    del self._keys[i]
                    self._save()
                    print(f"[ClientKeys] 删除 Key: {key_id}")
                    return True
        return False

    def get_key_by_id(self, key_id: str) -> Optional[ClientKey]:
        """根据 ID 获取 Key"""
        with self._lock:
            for client_key in self._keys:
                if client_key.id == key_id:
                    return client_key
        return None


# 全局实例
client_key_manager = ClientKeyManager()
