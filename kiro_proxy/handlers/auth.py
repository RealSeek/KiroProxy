"""管理员认证 API 处理"""
import time
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

from ..core.admin_auth import (
    verify_password, generate_token, add_token_record, find_valid_token,
    update_token_last_used, revoke_token, revoke_all_tokens, prune_expired_tokens,
    setup_admin_password, is_admin_password_configured, get_rate_limiter, hash_token
)
from ..core.persistence import load_config, save_config


def get_client_ip(request: Request) -> str:
    """获取客户端 IP 地址"""
    # 优先从 X-Forwarded-For 获取（代理环境）
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    # 其次从 X-Real-IP 获取
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # 最后使用直连 IP
    return request.client.host if request.client else "unknown"


async def login(request: Request):
    """
    POST /api/auth/login
    管理员登录
    """
    try:
        body = await request.json()
        password = body.get("password", "").strip()

        if not password:
            raise HTTPException(status_code=400, detail="密码不能为空")

        # 检查登录限速
        client_ip = get_client_ip(request)
        rate_limiter = get_rate_limiter()

        if not rate_limiter.is_allowed(client_ip):
            raise HTTPException(
                status_code=429,
                detail="登录尝试过于频繁，请稍后再试"
            )

        # 记录登录尝试
        rate_limiter.record_attempt(client_ip)

        # 加载配置
        config = load_config()

        # 检查是否已配置密码
        if not is_admin_password_configured():
            # 渐进式安全：未配置密码时，设置首个密码
            salt_b64, password_hash = setup_admin_password(password)
            config["admin_salt"] = salt_b64
            config["admin_password_hash"] = password_hash

            # 生成 Token
            token = generate_token()
            add_token_record(config, token)

            # 保存配置
            if not save_config(config):
                raise HTTPException(status_code=500, detail="保存配置失败")

            return JSONResponse({
                "success": True,
                "message": "管理员密码设置成功",
                "token": token,
                "first_setup": True
            })

        # 验证密码
        salt_b64 = config.get("admin_salt", "")
        expected_hash = config.get("admin_password_hash", "")

        if not verify_password(password, salt_b64, expected_hash):
            raise HTTPException(status_code=401, detail="密码错误")

        # 生成新 Token
        token = generate_token()
        add_token_record(config, token)

        # 清理过期 Token
        now = time.time()
        prune_expired_tokens(config, now)

        # 保存配置
        if not save_config(config):
            raise HTTPException(status_code=500, detail="保存配置失败")

        return JSONResponse({
            "success": True,
            "message": "登录成功",
            "token": token,
            "first_setup": False
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Auth] 登录错误: {e}")
        raise HTTPException(status_code=500, detail="登录失败")


async def logout(request: Request):
    """
    POST /api/auth/logout
    管理员登出（撤销当前 Token）
    """
    try:
        # 从 Authorization header 提取 Token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="缺少认证 Token")

        token = auth_header[7:]
        token_hash = hash_token(token)

        # 加载配置并撤销 Token
        config = load_config()
        revoked = revoke_token(config, token_hash)

        if revoked:
            save_config(config)

        return JSONResponse({
            "success": True,
            "message": "登出成功"
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Auth] 登出错误: {e}")
        raise HTTPException(status_code=500, detail="登出失败")


async def verify(request: Request):
    """
    GET /api/auth/verify
    验证 Token 有效性
    """
    try:
        # 从 Authorization header 提取 Token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="缺少认证 Token")

        token = auth_header[7:]
        token_hash = hash_token(token)

        # 验证 Token
        config = load_config()
        now = time.time()

        token_record = find_valid_token(config, token_hash, now)
        if not token_record:
            raise HTTPException(status_code=401, detail="Token 无效或已过期")

        # 更新最后使用时间
        update_token_last_used(config, token_hash, now)
        save_config(config)

        return JSONResponse({
            "success": True,
            "message": "Token 有效",
            "token_id": token_record.id,
            "expires_at": token_record.expires_at
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Auth] Token 验证错误: {e}")
        raise HTTPException(status_code=401, detail="Token 验证失败")


async def change_password(request: Request):
    """
    POST /api/auth/change-password
    修改管理员密码（可选功能）
    """
    try:
        body = await request.json()
        current_password = body.get("current_password", "").strip()
        new_password = body.get("new_password", "").strip()

        if not current_password or not new_password:
            raise HTTPException(status_code=400, detail="当前密码和新密码不能为空")

        if len(new_password) < 6:
            raise HTTPException(status_code=400, detail="新密码长度至少 6 位")

        # 加载配置
        config = load_config()

        # 验证当前密码
        salt_b64 = config.get("admin_salt", "")
        expected_hash = config.get("admin_password_hash", "")

        if not verify_password(current_password, salt_b64, expected_hash):
            raise HTTPException(status_code=401, detail="当前密码错误")

        # 设置新密码
        new_salt_b64, new_password_hash = setup_admin_password(new_password)
        config["admin_salt"] = new_salt_b64
        config["admin_password_hash"] = new_password_hash

        # 撤销所有现有 Token（强制重新登录）
        revoke_all_tokens(config)

        # 保存配置
        if not save_config(config):
            raise HTTPException(status_code=500, detail="保存配置失败")

        return JSONResponse({
            "success": True,
            "message": "密码修改成功，请重新登录"
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Auth] 修改密码错误: {e}")
        raise HTTPException(status_code=500, detail="修改密码失败")