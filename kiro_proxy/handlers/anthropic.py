"""Anthropic 协议处理 - /v1/messages"""
import json
import uuid
import time
import asyncio
import httpx
from fastapi import Request, HTTPException
from fastapi.responses import StreamingResponse

from ..config import KIRO_API_URL, map_model_name
from ..core import state, RetryableRequest, is_retryable_error, stats_manager, flow_monitor, TokenUsage
from ..core.state import RequestLog
from ..core.history_manager import HistoryManager, get_history_config, is_content_length_error, TruncateStrategy
from ..core.error_handler import classify_error, ErrorType, format_error_log
from ..core.rate_limiter import get_rate_limiter
from ..credential import quota_manager
from ..kiro_api import build_headers, build_kiro_request, parse_event_stream_full, parse_event_stream, is_quota_exceeded_error
from ..converters import (
    generate_session_id,
    convert_anthropic_tools_to_kiro,
    convert_anthropic_messages_to_kiro,
    convert_kiro_response_to_anthropic,
    extract_images_from_content,
    extract_thinking_from_content
)
from .websearch import has_web_search_tool, handle_web_search_request, filter_web_search_tools


def _extract_text_from_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            parts.append(_extract_text_from_content(item))
        return "".join(parts)
    if isinstance(content, dict):
        if "text" in content and isinstance(content.get("text"), str):
            return content["text"]
        if "content" in content:
            return _extract_text_from_content(content.get("content"))
    return ""


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return (len(text) + 3) // 4


def _count_tokens_from_messages(messages, system: str = "") -> int:
    total = _estimate_tokens(system) if system else 0
    for msg in messages or []:
        total += _estimate_tokens(_extract_text_from_content(msg.get("content")))
    return total


def _handle_kiro_error(status_code: int, error_text: str, account):
    """处理 Kiro API 错误，返回 (http_status, error_type, error_message)"""
    error = classify_error(status_code, error_text)
    
    # 打印友好的错误日志
    print(format_error_log(error, account.id if account else None))
    
    # 账号封禁 - 禁用账号
    if error.should_disable_account and account:
        account.enabled = False
        from ..credential import CredentialStatus
        account.status = CredentialStatus.SUSPENDED
        print(f"[Account] 账号 {account.id} 已被禁用 ({error.type.value})")

    # 配额超限 - 标记冷却
    elif error.type == ErrorType.RATE_LIMITED and account:
        account.mark_quota_exceeded(error.message[:100])

    # 映射错误类型
    error_type_map = {
        ErrorType.ACCOUNT_SUSPENDED: (403, "authentication_error"),
        ErrorType.QUOTA_EXHAUSTED: (402, "rate_limit_error"),
        ErrorType.RATE_LIMITED: (429, "rate_limit_error"),
        ErrorType.CONTENT_TOO_LONG: (400, "invalid_request_error"),
        ErrorType.AUTH_FAILED: (401, "authentication_error"),
        ErrorType.SERVICE_UNAVAILABLE: (503, "api_error"),
        ErrorType.MODEL_UNAVAILABLE: (503, "overloaded_error"),
        ErrorType.UNKNOWN: (500, "api_error"),
    }
    
    http_status, err_type = error_type_map.get(error.type, (500, "api_error"))
    return http_status, err_type, error.user_message, error


async def handle_count_tokens(request: Request):
    '''Handle /v1/messages/count_tokens requests.'''
    body = await request.json()
    messages = body.get("messages", [])
    system = body.get("system", "")
    if not messages and not system:
        raise HTTPException(400, "messages required")
    return {"input_tokens": _count_tokens_from_messages(messages, system)}


async def _call_kiro_for_summary(prompt: str, account, headers: dict) -> str:
    """调用 Kiro API 生成摘要（内部使用）"""
    kiro_request = build_kiro_request(prompt, "claude-haiku-4.5", [])  # 用快速模型生成摘要
    try:
        async with httpx.AsyncClient(verify=False, timeout=60) as client:
            resp = await client.post(KIRO_API_URL, json=kiro_request, headers=headers)
            if resp.status_code == 200:
                return parse_event_stream(resp.content)
    except Exception as e:
        print(f"[Summary] API 调用失败: {e}")
    return ""


async def handle_messages(request: Request):
    """处理 /v1/messages 请求"""
    start_time = time.time()
    log_id = uuid.uuid4().hex[:8]
    
    body = await request.json()
    model = map_model_name(body.get("model", "claude-sonnet-4"))
    messages = body.get("messages", [])
    system = body.get("system", "")
    stream = body.get("stream", False)
    tools = body.get("tools", [])
    thinking = body.get("thinking")
    output_config = body.get("output_config")

    if not messages:
        raise HTTPException(400, "messages required")

    session_id = generate_session_id(messages)
    account = state.get_available_account(session_id)

    if not account:
        raise HTTPException(503, "All accounts are rate limited or unavailable")

    # 创建 Flow 记录
    flow_id = flow_monitor.create_flow(
        protocol="anthropic",
        method="POST",
        path="/v1/messages",
        headers=dict(request.headers),
        body=body,
        account_id=account.id,
        account_name=account.name,
    )
    
    # 检查 token 是否即将过期，尝试刷新
    if account.is_token_expiring_soon(5):
        print(f"[Anthropic] Token 即将过期，尝试刷新: {account.id}")
        success, msg = await account.refresh_token()
        if not success:
            print(f"[Anthropic] Token 刷新失败: {msg}")
    
    token = account.get_token()
    if not token:
        flow_monitor.fail_flow(flow_id, "authentication_error", f"Failed to get token for account {account.name}")
        raise HTTPException(500, f"Failed to get token for account {account.name}")
    
    # 使用账号的动态 Machine ID（提前构建，供摘要使用）
    creds = account.get_credentials()
    headers = build_headers(
        token,
        machine_id=account.get_machine_id(),
        profile_arn=creds.profile_arn if creds else None,
        client_id=creds.client_id if creds else None
    )

    # 检查是否为 WebSearch 请求
    if has_web_search_tool(tools):
        # 过滤掉 web_search 工具，保留其他工具（如果有的话）
        other_tools = filter_web_search_tools(tools)
        if other_tools:
            # 如果还有其他工具，需要混合处理（暂不支持，先只处理 web_search）
            print(f"[Anthropic] 注意：请求中还有 {len(other_tools)} 个其他工具，将被忽略")

        return await handle_web_search_request(
            body=body,
            token=token,
            machine_id=account.get_machine_id(),
            model=model,
            profile_arn=creds.profile_arn if creds else None,
            client_id=creds.client_id if creds else None
        )

    # 限速检查
    rate_limiter = get_rate_limiter()
    can_request, wait_seconds, reason = rate_limiter.can_request(account.id)
    if not can_request:
        print(f"[Anthropic] 限速: {reason}")
        await asyncio.sleep(wait_seconds)
    
    # 转换消息格式
    user_content, history, tool_results = convert_anthropic_messages_to_kiro(messages, system, thinking, output_config)

    # 历史消息预处理
    history_manager = HistoryManager(get_history_config(), cache_key=session_id)
    
    # 检查是否需要智能摘要或错误重试预摘要
    async def api_caller(prompt: str) -> str:
        return await _call_kiro_for_summary(prompt, account, headers)
    if history_manager.should_summarize(history) or history_manager.should_pre_summary_for_error_retry(history, user_content):
        history = await history_manager.pre_process_async(history, user_content, api_caller)
    else:
        history = history_manager.pre_process(history, user_content)
    
    # 摘要/截断后再次修复历史交替和 toolUses/toolResults 配对
    from ..converters import fix_history_alternation
    history = fix_history_alternation(history)
    
    if history_manager.was_truncated:
        print(f"[Anthropic] {history_manager.truncate_info}")
    
    # 提取最后一条消息中的图片
    images = []
    if messages:
        last_msg = messages[-1]
        if last_msg.get("role") == "user":
            _, images = extract_images_from_content(last_msg.get("content", ""))
    
    # 构建 Kiro 请求
    kiro_tools = convert_anthropic_tools_to_kiro(tools) if tools else None

    # 验证当前请求的 tool_results 与 history 最后一条 assistant 的 toolUses 配对
    if tool_results and history:
        last_assistant = None
        for msg in reversed(history):
            if "assistantResponseMessage" in msg:
                last_assistant = msg["assistantResponseMessage"]
                break

        if last_assistant:
            tool_use_ids = set()
            for tu in last_assistant.get("toolUses", []) or []:
                tu_id = tu.get("toolUseId")
                if tu_id:
                    tool_use_ids.add(tu_id)

            if tool_use_ids:
                tool_results = [tr for tr in tool_results if tr.get("toolUseId") in tool_use_ids]
                # 反向清理孤立的 toolUse
                tool_result_ids = set(tr.get("toolUseId") for tr in tool_results if tr.get("toolUseId"))
                if tool_result_ids and len(tool_result_ids) < len(tool_use_ids):
                    paired = [tu for tu in last_assistant.get("toolUses", []) if tu.get("toolUseId") in tool_result_ids]
                    if paired:
                        last_assistant["toolUses"] = paired
                    else:
                        last_assistant.pop("toolUses", None)
            else:
                tool_results = []
        else:
            tool_results = []

    kiro_request = build_kiro_request(user_content, model, history, kiro_tools, images, tool_results)
    
    if stream:
        return await _handle_stream(kiro_request, headers, account, model, log_id, start_time, session_id, flow_id, history, user_content, kiro_tools, images, tool_results, history_manager)
    else:
        return await _handle_non_stream(kiro_request, headers, account, model, log_id, start_time, session_id, flow_id, history, user_content, kiro_tools, images, tool_results, history_manager)


async def _handle_stream(kiro_request, headers, account, model, log_id, start_time, session_id=None, flow_id=None, history=None, user_content="", kiro_tools=None, images=None, tool_results=None, history_manager=None):
    """Handle streaming responses with auto-retry on quota exceeded and network errors."""

    async def generate():
        nonlocal kiro_request, history
        current_account = account
        retry_count = 0
        max_retries = 2
        full_content = ""
        final_status_code = 200
        final_error_msg = None

        def record_log():
            """记录请求日志"""
            duration = (time.time() - start_time) * 1000
            state.add_log(RequestLog(
                id=log_id,
                timestamp=time.time(),
                method="POST",
                path="/v1/messages",
                model=model,
                account_id=current_account.id if current_account else None,
                status=final_status_code,
                duration_ms=duration,
                error=final_error_msg
            ))
        
        while retry_count <= max_retries:
            try:
                async with httpx.AsyncClient(verify=False, timeout=300) as client:
                    async with client.stream("POST", KIRO_API_URL, json=kiro_request, headers=headers) as response:
                        
                        # 处理配额超限
                        if response.status_code == 429 or is_quota_exceeded_error(response.status_code, ""):
                            current_account.mark_quota_exceeded("Rate limited (stream)")
                            
                            # 尝试切换账号
                            next_account = state.get_next_available_account(current_account.id)
                            if next_account and retry_count < max_retries:
                                print(f"[Stream] 配额超限，切换账号: {current_account.id} -> {next_account.id}")
                                current_account = next_account
                                token = current_account.get_token()
                                headers["Authorization"] = f"Bearer {token}"
                                retry_count += 1
                                continue
                            
                            if flow_id:
                                flow_monitor.fail_flow(flow_id, "rate_limit_error", "All accounts rate limited", 429)
                            final_status_code = 429
                            final_error_msg = "All accounts rate limited"
                            yield f'event: error\ndata: {{"type":"error","error":{{"type":"rate_limit_error","message":"All accounts rate limited"}}}}\n\n'
                            record_log()
                            return

                        # 处理可重试的服务端错误
                        if is_retryable_error(response.status_code):
                            if retry_count < max_retries:
                                print(f"[Stream] 服务端错误 {response.status_code}，重试 {retry_count + 1}/{max_retries}")
                                retry_count += 1
                                import asyncio
                                await asyncio.sleep(0.5 * (2 ** retry_count))
                                continue
                            if flow_id:
                                flow_monitor.fail_flow(flow_id, "api_error", "Server error after retries", response.status_code)
                            final_status_code = response.status_code
                            final_error_msg = "Server error after retries"
                            yield f'event: error\ndata: {{"type":"error","error":{{"type":"api_error","message":"Server error after retries"}}}}\n\n'
                            record_log()
                            return

                        if response.status_code != 200:
                            error_text = await response.aread()
                            error_str = error_text.decode()
                            print(f"[Stream] Kiro API error {response.status_code}: {error_str[:200]}")

                            # 使用统一的错误处理
                            http_status, error_type, error_msg, error_obj = _handle_kiro_error(
                                response.status_code, error_str, current_account
                            )
                            
                            # 账号封禁 - 尝试切换账号
                            if error_obj.should_switch_account:
                                next_account = state.get_next_available_account(current_account.id)
                                if next_account and retry_count < max_retries:
                                    print(f"[Stream] 切换账号: {current_account.id} -> {next_account.id}")
                                    current_account = next_account
                                    headers["Authorization"] = f"Bearer {current_account.get_token()}"
                                    retry_count += 1
                                    continue
                            
                            # 检查是否为内容长度超限错误，尝试截断重试
                            if error_obj.type == ErrorType.CONTENT_TOO_LONG:
                                history_chars, user_chars, total_chars = history_manager.estimate_request_chars(
                                    history, user_content
                                )
                                print(f"[Stream] 内容长度超限: history={history_chars} chars, user={user_chars} chars, total={total_chars} chars")
                                async def api_caller(prompt: str) -> str:
                                    return await _call_kiro_for_summary(prompt, current_account, headers)
                                truncated_history, should_retry = await history_manager.handle_length_error_async(
                                    history, retry_count, api_caller
                                )
                                if should_retry:
                                    print(f"[Stream] 内容长度超限，{history_manager.truncate_info}")
                                    history = truncated_history
                                    # 重新构建请求
                                    kiro_request = build_kiro_request(user_content, model, history, kiro_tools, images, tool_results)
                                    retry_count += 1
                                    continue
                            
                            if flow_id:
                                flow_monitor.fail_flow(flow_id, error_type, error_msg, response.status_code, error_str)
                            final_status_code = http_status
                            final_error_msg = error_msg
                            yield f'event: error\ndata: {{"type":"error","error":{{"type":"{error_type}","message":"{error_msg}"}}}}\n\n'
                            record_log()
                            return

                        # 标记开始流式传输
                        if flow_id:
                            flow_monitor.start_streaming(flow_id)

                        # 正常处理响应
                        msg_id = f"msg_{log_id}"
                        yield f'event: message_start\ndata: {{"type":"message_start","message":{{"id":"{msg_id}","type":"message","role":"assistant","content":[],"model":"{model}","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":0,"output_tokens":0}}}}}}\n\n'

                        full_response = b""
                        chunk_count = 0

                        # Thinking buffer 状态机
                        # phase: "buffering" -> "thinking" -> "text" 或 "buffering" -> "text"
                        phase = "buffering"
                        buffer = ""
                        thinking_content = ""
                        thinking_tail = ""  # 用于处理 </thinking> 跨 chunk 的情况
                        block_index = 0
                        # 缓冲阈值：如果累积了这么多字符还没看到 <thinking>，就认为没有 thinking
                        BUFFER_THRESHOLD = 200
                        # <thinking> 标签的所有可能前缀
                        THINKING_TAG = "<thinking>"
                        THINKING_PREFIXES = tuple(THINKING_TAG[:i] for i in range(1, len(THINKING_TAG)))
                        pending_bytes = b""  # 跨 chunk 缓冲
                        exception_stop_reason = None  # 流式 exception 检测

                        async for chunk in response.aiter_bytes():
                            chunk_count += 1
                            full_response += chunk

                            try:
                                # 将上次残留的不完整数据与当前 chunk 拼接
                                data = pending_bytes + chunk
                                pending_bytes = b""
                                pos = 0
                                while pos < len(data):
                                    if pos + 12 > len(data):
                                        # 不足以读取消息头，保留到下次
                                        pending_bytes = data[pos:]
                                        break
                                    total_len = int.from_bytes(data[pos:pos+4], 'big')
                                    if total_len == 0:
                                        break
                                    if total_len > len(data) - pos:
                                        # 消息不完整，保留到下次
                                        pending_bytes = data[pos:]
                                        break
                                    headers_len = int.from_bytes(data[pos+4:pos+8], 'big')
                                    payload_start = pos + 12 + headers_len
                                    payload_end = pos + total_len - 4

                                    if payload_start < payload_end:
                                        try:
                                            # 检测 exception 事件（从 headers 中）
                                            try:
                                                headers_str = data[pos + 12:pos + 12 + headers_len].decode('utf-8', errors='ignore')
                                                if 'exception' in headers_str and 'ContentLengthExceeded' in headers_str:
                                                    exception_stop_reason = "max_tokens"
                                            except:
                                                pass

                                            payload = json.loads(data[payload_start:payload_end].decode('utf-8'))
                                            content = None
                                            if 'assistantResponseEvent' in payload:
                                                content = payload['assistantResponseEvent'].get('content')
                                            elif 'content' in payload:
                                                content = payload['content']
                                            if content:
                                                full_content += content
                                                if flow_id:
                                                    flow_monitor.add_chunk(flow_id, content)

                                                if phase == "buffering":
                                                    buffer += content
                                                    # 检查是否包含完整的 <thinking> 开始标签
                                                    if "<thinking>" in buffer:
                                                        # 找到 thinking 标签
                                                        tag_pos = buffer.index("<thinking>")
                                                        before = buffer[:tag_pos].strip()

                                                        # 发送 thinking block start
                                                        yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":{block_index},"content_block":{{"type":"thinking","thinking":""}}}}\n\n'

                                                        # 检查是否有完整的 </thinking>
                                                        end_tag = "</thinking>"
                                                        end_pos = buffer.find(end_tag, tag_pos + 10)
                                                        if end_pos != -1:
                                                            # 完整的 thinking 块
                                                            thinking_content = buffer[tag_pos + 10:end_pos]
                                                            yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{block_index},"delta":{{"type":"thinking_delta","thinking":{json.dumps(thinking_content)}}}}}\n\n'
                                                            yield f'event: content_block_stop\ndata: {{"type":"content_block_stop","index":{block_index}}}\n\n'
                                                            block_index += 1

                                                            # 剩余内容作为 text 开始流式输出
                                                            remaining = buffer[end_pos + len(end_tag):]
                                                            phase = "text"
                                                            yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":{block_index},"content_block":{{"type":"text","text":""}}}}\n\n'
                                                            if remaining.strip():
                                                                yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{block_index},"delta":{{"type":"text_delta","text":{json.dumps(remaining)}}}}}\n\n'
                                                        else:
                                                            # thinking 还没结束，进入 thinking 阶段
                                                            thinking_content = buffer[tag_pos + 10:]
                                                            yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{block_index},"delta":{{"type":"thinking_delta","thinking":{json.dumps(thinking_content)}}}}}\n\n'
                                                            phase = "thinking"

                                                    elif len(buffer) > BUFFER_THRESHOLD or (buffer.strip() and not buffer.rstrip().endswith(THINKING_PREFIXES)):
                                                        # 超过阈值，或者缓冲区末尾不是 <thinking> 的前缀，直接作为 text 输出
                                                        phase = "text"
                                                        yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":{block_index},"content_block":{{"type":"text","text":""}}}}\n\n'
                                                        yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{block_index},"delta":{{"type":"text_delta","text":{json.dumps(buffer)}}}}}\n\n'

                                                elif phase == "thinking":
                                                    # 在 thinking 阶段，处理 </thinking> 可能跨 chunk 的情况
                                                    combined = thinking_tail + content
                                                    thinking_tail = ""
                                                    end_tag = "</thinking>"
                                                    if end_tag in combined:
                                                        end_pos = combined.index(end_tag)
                                                        # 发送 thinking 结束前的内容
                                                        before_end = combined[:end_pos]
                                                        if before_end:
                                                            thinking_content += before_end
                                                            yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{block_index},"delta":{{"type":"thinking_delta","thinking":{json.dumps(before_end)}}}}}\n\n'
                                                        yield f'event: content_block_stop\ndata: {{"type":"content_block_stop","index":{block_index}}}\n\n'
                                                        block_index += 1

                                                        # 切换到 text 阶段
                                                        phase = "text"
                                                        remaining = combined[end_pos + len(end_tag):]
                                                        yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":{block_index},"content_block":{{"type":"text","text":""}}}}\n\n'
                                                        if remaining.strip():
                                                            yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{block_index},"delta":{{"type":"text_delta","text":{json.dumps(remaining)}}}}}\n\n'
                                                    else:
                                                        # 检查末尾是否可能是 </thinking> 的部分前缀
                                                        safe_content = combined
                                                        for i in range(min(len(end_tag) - 1, len(combined)), 0, -1):
                                                            if combined.endswith(end_tag[:i]):
                                                                thinking_tail = combined[-i:]
                                                                safe_content = combined[:-i]
                                                                break
                                                        if safe_content:
                                                            thinking_content += safe_content
                                                            yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{block_index},"delta":{{"type":"thinking_delta","thinking":{json.dumps(safe_content)}}}}}\n\n'

                                                elif phase == "text":
                                                    # 正常流式输出 text
                                                    yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{block_index},"delta":{{"type":"text_delta","text":{json.dumps(content)}}}}}\n\n'

                                        except Exception:
                                            pass
                                    pos += total_len
                            except Exception:
                                pass

                        # 流结束后处理
                        result = parse_event_stream_full(full_response)

                        if phase == "buffering":
                            # 缓冲的内容还没发出去，用完整解析处理
                            parsed_content = "".join(result.get("content", []))
                            thinking_text, text_content = extract_thinking_from_content(parsed_content or buffer)

                            if thinking_text:
                                yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":{block_index},"content_block":{{"type":"thinking","thinking":""}}}}\n\n'
                                yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{block_index},"delta":{{"type":"thinking_delta","thinking":{json.dumps(thinking_text)}}}}}\n\n'
                                yield f'event: content_block_stop\ndata: {{"type":"content_block_stop","index":{block_index}}}\n\n'
                                block_index += 1

                            yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":{block_index},"content_block":{{"type":"text","text":""}}}}\n\n'
                            if text_content:
                                yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{block_index},"delta":{{"type":"text_delta","text":{json.dumps(text_content)}}}}}\n\n'

                        elif phase == "thinking":
                            # thinking 没有正常关闭，刷新残留的 tail
                            if thinking_tail:
                                yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{block_index},"delta":{{"type":"thinking_delta","thinking":{json.dumps(thinking_tail)}}}}}\n\n'
                            yield f'event: content_block_stop\ndata: {{"type":"content_block_stop","index":{block_index}}}\n\n'
                            block_index += 1
                            yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":{block_index},"content_block":{{"type":"text","text":""}}}}\n\n'

                        elif phase == "text":
                            # 正常 text 阶段，检查是否有未发送的内容
                            if not full_content:
                                parsed_content = "".join(result.get("content", []))
                                if parsed_content:
                                    full_content = parsed_content
                                    _, text_only = extract_thinking_from_content(full_content)
                                    yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{block_index},"delta":{{"type":"text_delta","text":{json.dumps(text_only)}}}}}\n\n'

                        # content_block_stop (text)
                        yield f'event: content_block_stop\ndata: {{"type":"content_block_stop","index":{block_index}}}\n\n'
                        block_index += 1

                        if result["tool_uses"]:
                            for tool_use in result["tool_uses"]:
                                yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":{block_index},"content_block":{{"type":"tool_use","id":"{tool_use["id"]}","name":"{tool_use["name"]}","input":{{}}}}}}\n\n'
                                yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{block_index},"delta":{{"type":"input_json_delta","partial_json":{json.dumps(json.dumps(tool_use["input"]))}}}}}\n\n'
                                yield f'event: content_block_stop\ndata: {{"type":"content_block_stop","index":{block_index}}}\n\n'
                                block_index += 1

                        stop_reason = exception_stop_reason or result["stop_reason"]
                        yield f'event: message_delta\ndata: {{"type":"message_delta","delta":{{"stop_reason":"{stop_reason}","stop_sequence":null}},"usage":{{"output_tokens":100}}}}\n\n'
                        yield f'event: message_stop\ndata: {{"type":"message_stop"}}\n\n'

                        # 完成 Flow
                        if flow_id:
                            flow_monitor.complete_flow(
                                flow_id,
                                status_code=200,
                                content=full_content,
                                tool_calls=result.get("tool_uses", []),
                                stop_reason=stop_reason,
                                usage=TokenUsage(
                                    input_tokens=result.get("input_tokens", 0),
                                    output_tokens=result.get("output_tokens", 0),
                                ),
                            )

                        current_account.request_count += 1
                        current_account.last_used = time.time()
                        get_rate_limiter().record_request(current_account.id)
                        record_log()
                        return

            except httpx.TimeoutException:
                if retry_count < max_retries:
                    print(f"[Stream] 请求超时，重试 {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    import asyncio
                    await asyncio.sleep(0.5 * (2 ** retry_count))
                    continue
                final_status_code = 408
                final_error_msg = "Request timeout after retries"
                if flow_id:
                    flow_monitor.fail_flow(flow_id, "timeout_error", "Request timeout after retries", 408)
                yield f'event: error\ndata: {{"type":"error","error":{{"type":"api_error","message":"Request timeout after retries"}}}}\n\n'
                record_log()
                return
            except httpx.ConnectError:
                if retry_count < max_retries:
                    print(f"[Stream] 连接错误，重试 {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    import asyncio
                    await asyncio.sleep(0.5 * (2 ** retry_count))
                    continue
                final_status_code = 502
                final_error_msg = "Connection error after retries"
                if flow_id:
                    flow_monitor.fail_flow(flow_id, "connection_error", "Connection error after retries", 502)
                yield f'event: error\ndata: {{"type":"error","error":{{"type":"api_error","message":"Connection error after retries"}}}}\n\n'
                record_log()
                return
            except Exception as e:
                # 检查是否为可重试的网络错误
                if is_retryable_error(None, e) and retry_count < max_retries:
                    print(f"[Stream] 网络错误，重试 {retry_count + 1}/{max_retries}: {type(e).__name__}")
                    retry_count += 1
                    import asyncio
                    await asyncio.sleep(0.5 * (2 ** retry_count))
                    continue
                final_status_code = 500
                final_error_msg = str(e)
                if flow_id:
                    flow_monitor.fail_flow(flow_id, "api_error", str(e), 500)
                yield f'event: error\ndata: {{"type":"error","error":{{"type":"api_error","message":"{str(e)}"}}}}\n\n'
                record_log()
                return

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _handle_non_stream(kiro_request, headers, account, model, log_id, start_time, session_id=None, flow_id=None, history=None, user_content="", kiro_tools=None, images=None, tool_results=None, history_manager=None):
    """Handle non-streaming responses with auto-retry on quota exceeded and network errors."""
    error_msg = None
    status_code = 200
    current_account = account
    max_retries = 2
    retry_ctx = RetryableRequest(max_retries=2)

    for retry in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(verify=False, timeout=300) as client:
                response = await client.post(KIRO_API_URL, json=kiro_request, headers=headers)
                status_code = response.status_code

                # 处理配额超限
                if response.status_code == 429 or is_quota_exceeded_error(response.status_code, response.text):
                    current_account.mark_quota_exceeded("Rate limited")
                    
                    # 尝试切换账号
                    next_account = state.get_next_available_account(current_account.id)
                    if next_account and retry < max_retries:
                        print(f"[NonStream] 配额超限，切换账号: {current_account.id} -> {next_account.id}")
                        current_account = next_account
                        token = current_account.get_token()
                        creds = current_account.get_credentials()
                        headers["Authorization"] = f"Bearer {token}"
                        continue
                    
                    if flow_id:
                        flow_monitor.fail_flow(flow_id, "rate_limit_error", "All accounts rate limited", 429)
                    raise HTTPException(429, "All accounts rate limited")

                # 处理可重试的服务端错误
                if is_retryable_error(response.status_code):
                    if retry < max_retries:
                        print(f"[NonStream] 服务端错误 {response.status_code}，重试 {retry + 1}/{max_retries}")
                        await retry_ctx.wait()
                        continue
                    if flow_id:
                        flow_monitor.fail_flow(flow_id, "api_error", f"Server error after {max_retries} retries", response.status_code)
                    raise HTTPException(response.status_code, f"Server error after {max_retries} retries")

                if response.status_code != 200:
                    error_msg = response.text
                    print(f"[NonStream] Kiro API Error {response.status_code}: {error_msg[:500]}")
                    
                    # 使用统一的错误处理
                    status, error_type, error_message, error_obj = _handle_kiro_error(
                        response.status_code, error_msg, current_account
                    )
                    
                    # 账号封禁或配额超限 - 尝试切换账号
                    if error_obj.should_switch_account:
                        next_account = state.get_next_available_account(current_account.id)
                        if next_account and retry < max_retries:
                            print(f"[NonStream] 切换账号: {current_account.id} -> {next_account.id}")
                            current_account = next_account
                            headers["Authorization"] = f"Bearer {current_account.get_token()}"
                            continue
                    
                    # 检查是否为内容长度超限错误，尝试截断重试
                    if error_obj.type == ErrorType.CONTENT_TOO_LONG and history_manager:
                        history_chars, user_chars, total_chars = history_manager.estimate_request_chars(
                            history, user_content
                        )
                        print(f"[NonStream] 内容长度超限: history={history_chars} chars, user={user_chars} chars, total={total_chars} chars")
                        async def api_caller(prompt: str) -> str:
                            return await _call_kiro_for_summary(prompt, current_account, headers)
                        truncated_history, should_retry = await history_manager.handle_length_error_async(
                            history, retry, api_caller
                        )
                        if should_retry:
                            print(f"[NonStream] 内容长度超限，{history_manager.truncate_info}")
                            history = truncated_history
                            kiro_request = build_kiro_request(user_content, model, history, kiro_tools, images, tool_results)
                            continue
                        else:
                            print(f"[NonStream] 内容长度超限但未重试: retry={retry}/{max_retries}")
                    
                    if flow_id:
                        flow_monitor.fail_flow(flow_id, error_type, error_message, status, error_msg)
                    raise HTTPException(status, error_message)

                result = parse_event_stream_full(response.content)
                current_account.request_count += 1
                current_account.last_used = time.time()
                get_rate_limiter().record_request(current_account.id)

                # 完成 Flow
                if flow_id:
                    flow_monitor.complete_flow(
                        flow_id,
                        status_code=200,
                        content=result.get("text", ""),
                        tool_calls=result.get("tool_uses", []),
                        stop_reason=result.get("stop_reason", ""),
                        usage=TokenUsage(
                            input_tokens=result.get("input_tokens", 0),
                            output_tokens=result.get("output_tokens", 0),
                        ),
                    )

                return convert_kiro_response_to_anthropic(result, model, f"msg_{log_id}")

        except HTTPException:
            raise
        except httpx.TimeoutException as e:
            error_msg = f"Request timeout: {e}"
            status_code = 408
            if retry < max_retries:
                print(f"[NonStream] 请求超时，重试 {retry + 1}/{max_retries}")
                await retry_ctx.wait()
                continue
            if flow_id:
                flow_monitor.fail_flow(flow_id, "timeout_error", "Request timeout after retries", 408)
            raise HTTPException(408, "Request timeout after retries")
        except httpx.ConnectError as e:
            error_msg = f"Connection error: {e}"
            status_code = 502
            if retry < max_retries:
                print(f"[NonStream] 连接错误，重试 {retry + 1}/{max_retries}")
                await retry_ctx.wait()
                continue
            if flow_id:
                flow_monitor.fail_flow(flow_id, "connection_error", "Connection error after retries", 502)
            raise HTTPException(502, "Connection error after retries")
        except Exception as e:
            error_msg = str(e)
            status_code = 500
            # 检查是否为可重试的网络错误
            if is_retryable_error(None, e) and retry < max_retries:
                print(f"[NonStream] 网络错误，重试 {retry + 1}/{max_retries}: {type(e).__name__}")
                await retry_ctx.wait()
                continue
            if flow_id:
                flow_monitor.fail_flow(flow_id, "api_error", str(e), 500)
            raise HTTPException(500, str(e))
        finally:
            if retry == max_retries or status_code == 200:
                duration = (time.time() - start_time) * 1000
                state.add_log(RequestLog(
                    id=log_id,
                    timestamp=time.time(),
                    method="POST",
                    path="/v1/messages",
                    model=model,
                    account_id=current_account.id if current_account else None,
                    status=status_code,
                    duration_ms=duration,
                    error=error_msg
                ))
                # 记录统计
                stats_manager.record_request(
                    account_id=current_account.id if current_account else "unknown",
                    model=model,
                    success=status_code == 200,
                    latency_ms=duration
                )
    
    raise HTTPException(503, "All retries exhausted")


# ==================== Claude Code 兼容端点 ====================
# /cc/v1/messages - 流式响应会等待 contextUsageEvent 后再发送 message_start
# message_start 中的 input_tokens 是从 contextUsageEvent 计算的准确值

CONTEXT_WINDOW_SIZE = 200_000  # 上下文窗口大小（200k tokens）
PING_INTERVAL_SECS = 25  # Ping 保活间隔


async def handle_messages_cc(request: Request):
    """处理 /cc/v1/messages 请求 - Claude Code 兼容端点

    与 /v1/messages 的区别：
    - 流式响应会等待 kiro 端返回 contextUsageEvent 后再发送 message_start
    - message_start 中的 input_tokens 是从 contextUsageEvent 计算的准确值
    """
    start_time = time.time()
    log_id = uuid.uuid4().hex[:8]

    body = await request.json()
    model = map_model_name(body.get("model", "claude-sonnet-4"))
    messages = body.get("messages", [])
    system = body.get("system", "")
    stream = body.get("stream", False)
    tools = body.get("tools", [])
    thinking = body.get("thinking")
    output_config = body.get("output_config")

    if not messages:
        raise HTTPException(400, "messages required")

    session_id = generate_session_id(messages)
    account = state.get_available_account(session_id)

    if not account:
        raise HTTPException(503, "All accounts are rate limited or unavailable")

    # 创建 Flow 记录
    flow_id = flow_monitor.create_flow(
        protocol="anthropic",
        method="POST",
        path="/cc/v1/messages",
        headers=dict(request.headers),
        body=body,
        account_id=account.id,
        account_name=account.name,
    )

    # 检查 token 是否即将过期，尝试刷新
    if account.is_token_expiring_soon(5):
        print(f"[CC/Anthropic] Token 即将过期，尝试刷新: {account.id}")
        success, msg = await account.refresh_token()
        if not success:
            print(f"[CC/Anthropic] Token 刷新失败: {msg}")

    token = account.get_token()
    if not token:
        flow_monitor.fail_flow(flow_id, "authentication_error", f"Failed to get token for account {account.name}")
        raise HTTPException(500, f"Failed to get token for account {account.name}")

    # 使用账号的动态 Machine ID
    creds = account.get_credentials()
    headers = build_headers(
        token,
        machine_id=account.get_machine_id(),
        profile_arn=creds.profile_arn if creds else None,
        client_id=creds.client_id if creds else None
    )

    # 检查是否为 WebSearch 请求
    if has_web_search_tool(tools):
        other_tools = filter_web_search_tools(tools)
        if other_tools:
            print(f"[CC/Anthropic] 注意：请求中还有 {len(other_tools)} 个其他工具，将被忽略")

        return await handle_web_search_request(
            body=body,
            token=token,
            machine_id=account.get_machine_id(),
            model=model,
            profile_arn=creds.profile_arn if creds else None,
            client_id=creds.client_id if creds else None
        )

    # 限速检查
    rate_limiter = get_rate_limiter()
    can_request, wait_seconds, reason = rate_limiter.can_request(account.id)
    if not can_request:
        print(f"[CC/Anthropic] 限速: {reason}")
        await asyncio.sleep(wait_seconds)

    # 转换消息格式
    user_content, history, tool_results = convert_anthropic_messages_to_kiro(messages, system, thinking, output_config)

    # 历史消息预处理
    # Claude Code 客户端拥有自己的上下文管理能力（基于准确的 input_tokens）
    # 因此 /cc/v1 端点禁用后端自动截断/重试，让长度超限错误直接透传给客户端
    from ..core.history_manager import HistoryConfig
    cc_history_config = HistoryConfig(strategies=[])  # 禁用所有截断策略
    history_manager = HistoryManager(cc_history_config, cache_key=session_id)

    from ..converters import fix_history_alternation
    history = fix_history_alternation(history)

    if history_manager.was_truncated:
        print(f"[CC/Anthropic] {history_manager.truncate_info}")

    # 提取最后一条消息中的图片
    images = []
    if messages:
        last_msg = messages[-1]
        if last_msg.get("role") == "user":
            _, images = extract_images_from_content(last_msg.get("content", ""))

    # 构建 Kiro 请求
    kiro_tools = convert_anthropic_tools_to_kiro(tools) if tools else None

    # 调试：打印工具信息
    if tools:
        print(f"[CC/Anthropic] 收到 {len(tools)} 个工具，转换后 {len(kiro_tools) if kiro_tools else 0} 个")
        for i, t in enumerate(tools[:5]):  # 只打印前5个
            print(f"[CC/Anthropic]   工具 {i+1}: type={t.get('type', '')}, name={t.get('name', '')}")

    # 验证当前请求的 tool_results 与 history 最后一条 assistant 的 toolUses 配对
    if tool_results and history:
        last_assistant = None
        for msg in reversed(history):
            if "assistantResponseMessage" in msg:
                last_assistant = msg["assistantResponseMessage"]
                break

        if last_assistant:
            tool_use_ids = set()
            for tu in last_assistant.get("toolUses", []) or []:
                tu_id = tu.get("toolUseId")
                if tu_id:
                    tool_use_ids.add(tu_id)

            if tool_use_ids:
                tool_results = [tr for tr in tool_results if tr.get("toolUseId") in tool_use_ids]
                tool_result_ids = set(tr.get("toolUseId") for tr in tool_results if tr.get("toolUseId"))
                if tool_result_ids and len(tool_result_ids) < len(tool_use_ids):
                    paired = [tu for tu in last_assistant.get("toolUses", []) if tu.get("toolUseId") in tool_result_ids]
                    if paired:
                        last_assistant["toolUses"] = paired
                    else:
                        last_assistant.pop("toolUses", None)
            else:
                tool_results = []
        else:
            tool_results = []

    kiro_request = build_kiro_request(user_content, model, history, kiro_tools, images, tool_results)

    if stream:
        return await _handle_stream_cc(kiro_request, headers, account, model, log_id, start_time, session_id, flow_id, history, user_content, kiro_tools, images, tool_results, history_manager)
    else:
        return await _handle_non_stream_cc(kiro_request, headers, account, model, log_id, start_time, session_id, flow_id, history, user_content, kiro_tools, images, tool_results, history_manager)


async def _handle_stream_cc(kiro_request, headers, account, model, log_id, start_time, session_id=None, flow_id=None, history=None, user_content="", kiro_tools=None, images=None, tool_results=None, history_manager=None):
    """处理流式响应 - 缓冲模式

    与普通流式处理不同，此函数会：
    1. 缓冲所有事件直到流结束
    2. 期间只发送 ping 保活信号
    3. 流结束后，用从 contextUsageEvent 计算的正确 input_tokens 生成 message_start
    4. 一次性发送所有事件
    """

    async def generate():
        nonlocal kiro_request, history
        current_account = account
        retry_count = 0
        max_retries = 2
        final_status_code = 200
        final_error_msg = None

        def record_log():
            """记录请求日志"""
            duration = (time.time() - start_time) * 1000
            state.add_log(RequestLog(
                id=log_id,
                timestamp=time.time(),
                method="POST",
                path="/cc/v1/messages",
                model=model,
                account_id=current_account.id if current_account else None,
                status=final_status_code,
                duration_ms=duration,
                error=final_error_msg
            ))

        while retry_count <= max_retries:
            try:
                async with httpx.AsyncClient(verify=False, timeout=300) as client:
                    # 发送 ping 保活的任务
                    ping_task = None
                    should_stop_ping = False

                    async def send_pings():
                        """发送 ping 保活信号"""
                        nonlocal should_stop_ping
                        while not should_stop_ping:
                            await asyncio.sleep(PING_INTERVAL_SECS)
                            if not should_stop_ping:
                                yield 'event: ping\ndata: {"type": "ping"}\n\n'

                    async with client.stream("POST", KIRO_API_URL, json=kiro_request, headers=headers) as response:

                        # 处理配额超限
                        if response.status_code == 429 or is_quota_exceeded_error(response.status_code, ""):
                            current_account.mark_quota_exceeded("Rate limited (stream)")

                            next_account = state.get_next_available_account(current_account.id)
                            if next_account and retry_count < max_retries:
                                print(f"[CC/Stream] 配额超限，切换账号: {current_account.id} -> {next_account.id}")
                                current_account = next_account
                                token = current_account.get_token()
                                headers["Authorization"] = f"Bearer {token}"
                                retry_count += 1
                                continue

                            if flow_id:
                                flow_monitor.fail_flow(flow_id, "rate_limit_error", "All accounts rate limited", 429)
                            final_status_code = 429
                            final_error_msg = "All accounts rate limited"
                            yield f'event: error\ndata: {{"type":"error","error":{{"type":"rate_limit_error","message":"All accounts rate limited"}}}}\n\n'
                            record_log()
                            return

                        # 处理可重试的服务端错误
                        if is_retryable_error(response.status_code):
                            if retry_count < max_retries:
                                print(f"[CC/Stream] 服务端错误 {response.status_code}，重试 {retry_count + 1}/{max_retries}")
                                retry_count += 1
                                await asyncio.sleep(0.5 * (2 ** retry_count))
                                continue
                            if flow_id:
                                flow_monitor.fail_flow(flow_id, "api_error", "Server error after retries", response.status_code)
                            final_status_code = response.status_code
                            final_error_msg = "Server error after retries"
                            yield f'event: error\ndata: {{"type":"error","error":{{"type":"api_error","message":"Server error after retries"}}}}\n\n'
                            record_log()
                            return

                        if response.status_code != 200:
                            error_text = await response.aread()
                            error_str = error_text.decode()
                            print(f"[CC/Stream] Kiro API error {response.status_code}: {error_str[:200]}")

                            http_status, error_type, error_msg, error_obj = _handle_kiro_error(
                                response.status_code, error_str, current_account
                            )

                            if error_obj.should_switch_account:
                                next_account = state.get_next_available_account(current_account.id)
                                if next_account and retry_count < max_retries:
                                    print(f"[CC/Stream] 切换账号: {current_account.id} -> {next_account.id}")
                                    current_account = next_account
                                    headers["Authorization"] = f"Bearer {current_account.get_token()}"
                                    retry_count += 1
                                    continue

                            if error_obj.type == ErrorType.CONTENT_TOO_LONG:
                                # /cc 端点：返回 stop_reason="max_tokens" 让 Claude Code 触发自动压缩
                                # 而不是返回错误，这样客户端可以识别上下文已满并自动处理
                                history_chars, user_chars, total_chars = history_manager.estimate_request_chars(
                                    history, user_content
                                )
                                print(f"[CC/Stream] 内容长度超限: history={history_chars} chars, user={user_chars} chars, total={total_chars} chars")
                                print(f"[CC/Stream] 返回 stop_reason=max_tokens 以触发 Claude Code 自动压缩")

                                # 估算 input_tokens（基于字符数，约 4 字符/token）
                                estimated_input_tokens = total_chars // 4
                                # 确保接近上下文窗口限制，触发压缩
                                estimated_input_tokens = max(estimated_input_tokens, CONTEXT_WINDOW_SIZE - 1000)

                                msg_id = f"msg_{log_id}"

                                # 发送完整的流结束序列，stop_reason="max_tokens"
                                yield f'event: message_start\ndata: {{"type":"message_start","message":{{"id":"{msg_id}","type":"message","role":"assistant","content":[],"model":"{model}","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":{estimated_input_tokens},"output_tokens":1}}}}}}\n\n'
                                yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":0,"content_block":{{"type":"text","text":""}}}}\n\n'
                                yield f'event: content_block_stop\ndata: {{"type":"content_block_stop","index":0}}\n\n'
                                yield f'event: message_delta\ndata: {{"type":"message_delta","delta":{{"stop_reason":"max_tokens","stop_sequence":null}},"usage":{{"input_tokens":{estimated_input_tokens},"output_tokens":1}}}}\n\n'
                                yield f'event: message_stop\ndata: {{"type":"message_stop"}}\n\n'

                                if flow_id:
                                    flow_monitor.complete_flow(
                                        flow_id,
                                        status_code=200,
                                        content="",
                                        tool_calls=[],
                                        stop_reason="max_tokens",
                                        usage=TokenUsage(input_tokens=estimated_input_tokens, output_tokens=1),
                                    )
                                record_log()
                                return

                            if flow_id:
                                flow_monitor.fail_flow(flow_id, error_type, error_msg, response.status_code, error_str)
                            final_status_code = http_status
                            final_error_msg = error_msg
                            yield f'event: error\ndata: {{"type":"error","error":{{"type":"{error_type}","message":"{error_msg}"}}}}\n\n'
                            record_log()
                            return

                        # 标记开始流式传输
                        if flow_id:
                            flow_monitor.start_streaming(flow_id)

                        # 缓冲模式：先读取完整响应，期间发送 ping 保活
                        full_response = b""
                        last_ping_time = time.time()

                        async for chunk in response.aiter_bytes():
                            full_response += chunk

                            # 每 PING_INTERVAL_SECS 秒发送一次 ping
                            current_time = time.time()
                            if current_time - last_ping_time >= PING_INTERVAL_SECS:
                                yield 'event: ping\ndata: {"type": "ping"}\n\n'
                                last_ping_time = current_time

                        # 解析完整响应
                        result = parse_event_stream_full(full_response)
                        full_content = "".join(result.get("content", []))

                        # 获取从 contextUsageEvent 计算的 input_tokens
                        input_tokens = result.get("input_tokens") or 0

                        # 提取 thinking 内容
                        thinking_text, text_content = extract_thinking_from_content(full_content)

                        # 估算 output_tokens（排除 thinking 内容，只计算 text + tool_use）
                        # Claude Code 有 CLAUDE_CODE_MAX_OUTPUT_TOKENS 限制（默认 32000），
                        # thinking tokens 不应计入 output_tokens，否则会触发该限制
                        output_tokens = _estimate_tokens(text_content)
                        for tool_use in result.get("tool_uses", []):
                            output_tokens += _estimate_tokens(json.dumps(tool_use.get("input", {})))

                        # 现在一次性发送所有事件
                        msg_id = f"msg_{log_id}"
                        block_index = 0

                        # message_start - 使用准确的 input_tokens
                        yield f'event: message_start\ndata: {{"type":"message_start","message":{{"id":"{msg_id}","type":"message","role":"assistant","content":[],"model":"{model}","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":{input_tokens},"output_tokens":1}}}}}}\n\n'

                        # thinking block (如果有)
                        if thinking_text:
                            yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":{block_index},"content_block":{{"type":"thinking","thinking":""}}}}\n\n'
                            yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{block_index},"delta":{{"type":"thinking_delta","thinking":{json.dumps(thinking_text)}}}}}\n\n'
                            yield f'event: content_block_stop\ndata: {{"type":"content_block_stop","index":{block_index}}}\n\n'
                            block_index += 1

                        # content_block_start (text)
                        yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":{block_index},"content_block":{{"type":"text","text":""}}}}\n\n'

                        # content_block_delta (text)
                        if text_content:
                            yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{block_index},"delta":{{"type":"text_delta","text":{json.dumps(text_content)}}}}}\n\n'

                        # content_block_stop (text)
                        yield f'event: content_block_stop\ndata: {{"type":"content_block_stop","index":{block_index}}}\n\n'
                        block_index += 1

                        # tool_use blocks
                        if result["tool_uses"]:
                            for tool_use in result["tool_uses"]:
                                yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":{block_index},"content_block":{{"type":"tool_use","id":"{tool_use["id"]}","name":"{tool_use["name"]}","input":{{}}}}}}\n\n'
                                yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{block_index},"delta":{{"type":"input_json_delta","partial_json":{json.dumps(json.dumps(tool_use["input"]))}}}}}\n\n'
                                yield f'event: content_block_stop\ndata: {{"type":"content_block_stop","index":{block_index}}}\n\n'
                                block_index += 1

                        stop_reason = result["stop_reason"]

                        # message_delta - 使用准确的 token 数
                        yield f'event: message_delta\ndata: {{"type":"message_delta","delta":{{"stop_reason":"{stop_reason}","stop_sequence":null}},"usage":{{"input_tokens":{input_tokens},"output_tokens":{output_tokens}}}}}\n\n'
                        yield f'event: message_stop\ndata: {{"type":"message_stop"}}\n\n'

                        # 完成 Flow
                        if flow_id:
                            flow_monitor.complete_flow(
                                flow_id,
                                status_code=200,
                                content=full_content,
                                tool_calls=result.get("tool_uses", []),
                                stop_reason=stop_reason,
                                usage=TokenUsage(
                                    input_tokens=input_tokens,
                                    output_tokens=output_tokens,
                                ),
                            )

                        current_account.request_count += 1
                        current_account.last_used = time.time()
                        get_rate_limiter().record_request(current_account.id)
                        record_log()
                        return

            except httpx.TimeoutException:
                if retry_count < max_retries:
                    print(f"[CC/Stream] 请求超时，重试 {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    await asyncio.sleep(0.5 * (2 ** retry_count))
                    continue
                final_status_code = 408
                final_error_msg = "Request timeout after retries"
                if flow_id:
                    flow_monitor.fail_flow(flow_id, "timeout_error", "Request timeout after retries", 408)
                yield f'event: error\ndata: {{"type":"error","error":{{"type":"api_error","message":"Request timeout after retries"}}}}\n\n'
                record_log()
                return
            except httpx.ConnectError:
                if retry_count < max_retries:
                    print(f"[CC/Stream] 连接错误，重试 {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    await asyncio.sleep(0.5 * (2 ** retry_count))
                    continue
                final_status_code = 502
                final_error_msg = "Connection error after retries"
                if flow_id:
                    flow_monitor.fail_flow(flow_id, "connection_error", "Connection error after retries", 502)
                yield f'event: error\ndata: {{"type":"error","error":{{"type":"api_error","message":"Connection error after retries"}}}}\n\n'
                record_log()
                return
            except Exception as e:
                if is_retryable_error(None, e) and retry_count < max_retries:
                    print(f"[CC/Stream] 网络错误，重试 {retry_count + 1}/{max_retries}: {type(e).__name__}")
                    retry_count += 1
                    await asyncio.sleep(0.5 * (2 ** retry_count))
                    continue
                final_status_code = 500
                final_error_msg = str(e)
                if flow_id:
                    flow_monitor.fail_flow(flow_id, "api_error", str(e), 500)
                yield f'event: error\ndata: {{"type":"error","error":{{"type":"api_error","message":"{str(e)}"}}}}\n\n'
                record_log()
                return

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _handle_non_stream_cc(kiro_request, headers, account, model, log_id, start_time, session_id=None, flow_id=None, history=None, user_content="", kiro_tools=None, images=None, tool_results=None, history_manager=None):
    """处理非流式响应 - 使用 contextUsageEvent 计算准确的 input_tokens"""
    error_msg = None
    status_code = 200
    current_account = account
    max_retries = 2
    retry_ctx = RetryableRequest(max_retries=2)

    for retry in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(verify=False, timeout=300) as client:
                response = await client.post(KIRO_API_URL, json=kiro_request, headers=headers)
                status_code = response.status_code

                # 处理配额超限
                if response.status_code == 429 or is_quota_exceeded_error(response.status_code, response.text):
                    current_account.mark_quota_exceeded("Rate limited")

                    next_account = state.get_next_available_account(current_account.id)
                    if next_account and retry < max_retries:
                        print(f"[CC/NonStream] 配额超限，切换账号: {current_account.id} -> {next_account.id}")
                        current_account = next_account
                        token = current_account.get_token()
                        creds = current_account.get_credentials()
                        headers["Authorization"] = f"Bearer {token}"
                        continue

                    if flow_id:
                        flow_monitor.fail_flow(flow_id, "rate_limit_error", "All accounts rate limited", 429)
                    raise HTTPException(429, "All accounts rate limited")

                # 处理可重试的服务端错误
                if is_retryable_error(response.status_code):
                    if retry < max_retries:
                        print(f"[CC/NonStream] 服务端错误 {response.status_code}，重试 {retry + 1}/{max_retries}")
                        await retry_ctx.wait()
                        continue
                    if flow_id:
                        flow_monitor.fail_flow(flow_id, "api_error", f"Server error after {max_retries} retries", response.status_code)
                    raise HTTPException(response.status_code, f"Server error after {max_retries} retries")

                if response.status_code != 200:
                    error_msg = response.text
                    print(f"[CC/NonStream] Kiro API Error {response.status_code}: {error_msg[:500]}")

                    status, error_type, error_message, error_obj = _handle_kiro_error(
                        response.status_code, error_msg, current_account
                    )

                    if error_obj.should_switch_account:
                        next_account = state.get_next_available_account(current_account.id)
                        if next_account and retry < max_retries:
                            print(f"[CC/NonStream] 切换账号: {current_account.id} -> {next_account.id}")
                            current_account = next_account
                            headers["Authorization"] = f"Bearer {current_account.get_token()}"
                            continue

                    if error_obj.type == ErrorType.CONTENT_TOO_LONG and history_manager:
                        # /cc 端点：返回 stop_reason="max_tokens" 让 Claude Code 触发自动压缩
                        history_chars, user_chars, total_chars = history_manager.estimate_request_chars(
                            history, user_content
                        )
                        print(f"[CC/NonStream] 内容长度超限: history={history_chars} chars, user={user_chars} chars, total={total_chars} chars")
                        print(f"[CC/NonStream] 返回 stop_reason=max_tokens 以触发 Claude Code 自动压缩")

                        # 估算 input_tokens
                        estimated_input_tokens = total_chars // 4
                        estimated_input_tokens = max(estimated_input_tokens, CONTEXT_WINDOW_SIZE - 1000)

                        if flow_id:
                            flow_monitor.complete_flow(
                                flow_id,
                                status_code=200,
                                content="",
                                tool_calls=[],
                                stop_reason="max_tokens",
                                usage=TokenUsage(input_tokens=estimated_input_tokens, output_tokens=1),
                            )

                        # 返回带有 stop_reason="max_tokens" 的正常响应
                        return {
                            "id": f"msg_{log_id}",
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model,
                            "stop_reason": "max_tokens",
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": estimated_input_tokens,
                                "output_tokens": 1
                            }
                        }

                    if flow_id:
                        flow_monitor.fail_flow(flow_id, error_type, error_message, status, error_msg)
                    raise HTTPException(status, error_message)

                result = parse_event_stream_full(response.content)
                current_account.request_count += 1
                current_account.last_used = time.time()
                get_rate_limiter().record_request(current_account.id)

                # 获取从 contextUsageEvent 计算的 input_tokens
                input_tokens = result.get("input_tokens") or 0

                # 估算 output_tokens（排除 thinking 内容，避免触发 Claude Code 的 output token 限制）
                full_content = "".join(result.get("content", []))
                _, text_only = extract_thinking_from_content(full_content)
                output_tokens = _estimate_tokens(text_only)
                for tool_use in result.get("tool_uses", []):
                    output_tokens += _estimate_tokens(json.dumps(tool_use.get("input", {})))

                # 完成 Flow
                if flow_id:
                    flow_monitor.complete_flow(
                        flow_id,
                        status_code=200,
                        content=full_content,
                        tool_calls=result.get("tool_uses", []),
                        stop_reason=result.get("stop_reason", ""),
                        usage=TokenUsage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                        ),
                    )

                # 构建响应，使用准确的 input_tokens
                return _build_anthropic_response_cc(result, model, f"msg_{log_id}", input_tokens, output_tokens)

        except HTTPException:
            raise
        except httpx.TimeoutException as e:
            error_msg = f"Request timeout: {e}"
            status_code = 408
            if retry < max_retries:
                print(f"[CC/NonStream] 请求超时，重试 {retry + 1}/{max_retries}")
                await retry_ctx.wait()
                continue
            if flow_id:
                flow_monitor.fail_flow(flow_id, "timeout_error", "Request timeout after retries", 408)
            raise HTTPException(408, "Request timeout after retries")
        except httpx.ConnectError as e:
            error_msg = f"Connection error: {e}"
            status_code = 502
            if retry < max_retries:
                print(f"[CC/NonStream] 连接错误，重试 {retry + 1}/{max_retries}")
                await retry_ctx.wait()
                continue
            if flow_id:
                flow_monitor.fail_flow(flow_id, "connection_error", "Connection error after retries", 502)
            raise HTTPException(502, "Connection error after retries")
        except Exception as e:
            error_msg = str(e)
            status_code = 500
            if is_retryable_error(None, e) and retry < max_retries:
                print(f"[CC/NonStream] 网络错误，重试 {retry + 1}/{max_retries}: {type(e).__name__}")
                await retry_ctx.wait()
                continue
            if flow_id:
                flow_monitor.fail_flow(flow_id, "api_error", str(e), 500)
            raise HTTPException(500, str(e))
        finally:
            if retry == max_retries or status_code == 200:
                duration = (time.time() - start_time) * 1000
                state.add_log(RequestLog(
                    id=log_id,
                    timestamp=time.time(),
                    method="POST",
                    path="/cc/v1/messages",
                    model=model,
                    account_id=current_account.id if current_account else None,
                    status=status_code,
                    duration_ms=duration,
                    error=error_msg
                ))
                stats_manager.record_request(
                    account_id=current_account.id if current_account else "unknown",
                    model=model,
                    success=status_code == 200,
                    latency_ms=duration
                )

    raise HTTPException(503, "All retries exhausted")


def _build_anthropic_response_cc(result: dict, model: str, msg_id: str, input_tokens: int, output_tokens: int) -> dict:
    """构建 Anthropic 格式响应 - 使用准确的 token 数"""
    content = []

    # 文本内容
    text = "".join(result.get("content", []))

    # 提取 thinking 内容
    thinking_text, text = extract_thinking_from_content(text)
    if thinking_text:
        content.append({"type": "thinking", "thinking": thinking_text})

    if text:
        content.append({"type": "text", "text": text})

    # 工具调用
    for tool_use in result.get("tool_uses", []):
        content.append({
            "type": "tool_use",
            "id": tool_use["id"],
            "name": tool_use["name"],
            "input": tool_use["input"]
        })

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": result.get("stop_reason", "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    }
