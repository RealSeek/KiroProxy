"""Kiro Provider"""
import json
import os
import uuid
from typing import Dict, Any, List, Optional, Tuple

from .base import BaseProvider
from ..credential import (
    KiroCredentials, TokenRefresher,
    generate_machine_id, get_kiro_version, get_system_info
)


def _load_debug_usage_config() -> Tuple[bool, int]:
    enabled = os.getenv("KIRO_DEBUG_USAGE", "").strip().lower() in {"1", "true", "yes", "y"}
    max_chars = 2000
    raw_max = os.getenv("KIRO_DEBUG_USAGE_MAX_CHARS", "").strip()
    if raw_max:
        try:
            max_chars = max(200, int(raw_max))
        except ValueError:
            max_chars = 2000
    return enabled, max_chars


_DEBUG_USAGE_ENABLED, _DEBUG_USAGE_MAX_CHARS = _load_debug_usage_config()


def _truncate_debug_payload(payload: Any, max_chars: int) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = str(payload)
    if len(text) > max_chars:
        return f"{text[:max_chars]}...<truncated>"
    return text


def _extract_usage_fields(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    usage_fields: Dict[str, Any] = {}
    for key, value in obj.items():
        if isinstance(key, str):
            key_lower = key.lower()
            if "token" in key_lower or "usage" in key_lower:
                usage_fields[key] = value
    return usage_fields or None


def _debug_usage_log(tag: str, payload: Any) -> None:
    if not _DEBUG_USAGE_ENABLED:
        return
    text = _truncate_debug_payload(payload, _DEBUG_USAGE_MAX_CHARS)
    print(f"[Kiro][UsageDebug] {tag}: {text}")


class KiroProvider(BaseProvider):
    """Kiro/CodeWhisperer Provider"""

    API_URL_TEMPLATE = "https://q.{region}.amazonaws.com/generateAssistantResponse"
    MODELS_URL_TEMPLATE = "https://q.{region}.amazonaws.com/ListAvailableModels"

    def __init__(self, credentials: Optional[KiroCredentials] = None):
        self.credentials = credentials
        self._machine_id: Optional[str] = None

    def _get_region(self) -> str:
        if self.credentials and self.credentials.region:
            return self.credentials.region
        return "us-east-1"

    @property
    def name(self) -> str:
        return "kiro"

    @property
    def api_url(self) -> str:
        return self.API_URL_TEMPLATE.format(region=self._get_region())

    @property
    def models_url(self) -> str:
        return self.MODELS_URL_TEMPLATE.format(region=self._get_region())
    
    def get_machine_id(self) -> str:
        """获取基于凭证的 Machine ID"""
        if self._machine_id:
            return self._machine_id
        
        if self.credentials:
            self._machine_id = generate_machine_id(
                self.credentials.profile_arn,
                self.credentials.client_id
            )
        else:
            self._machine_id = generate_machine_id()
        
        return self._machine_id
    
    def build_headers(
        self, 
        token: str, 
        agent_mode: str = "vibe",
        **kwargs
    ) -> Dict[str, str]:
        """构建 Kiro API 请求头"""
        machine_id = kwargs.get("machine_id") or self.get_machine_id()
        kiro_version = get_kiro_version()
        os_name, node_version = get_system_info()
        
        return {
            "content-type": "application/json",
            "x-amzn-codewhisperer-optout": "true",
            "x-amzn-kiro-agent-mode": agent_mode,
            "x-amz-user-agent": f"aws-sdk-js/1.0.27 KiroIDE-{kiro_version}-{machine_id}",
            "user-agent": f"aws-sdk-js/1.0.27 ua/2.1 os/{os_name} lang/js md/nodejs#{node_version} api/codewhispererstreaming#1.0.27 m/E KiroIDE-{kiro_version}-{machine_id}",
            "amz-sdk-invocation-id": str(uuid.uuid4()),
            "amz-sdk-request": "attempt=1; max=3",
            "Authorization": f"Bearer {token}",
            "Connection": "close",
        }
    
    def build_request(
        self,
        messages: list = None,
        model: str = "claude-sonnet-4",
        user_content: str = "",
        history: List[dict] = None,
        tools: List[dict] = None,
        images: List[dict] = None,
        tool_results: List[dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """构建 Kiro API 请求体"""
        conversation_id = str(uuid.uuid4())
        
        # 确保 content 不为空
        if not user_content:
            user_content = "Continue"
        
        user_input_message = {
            "content": user_content,
            "modelId": model,
            "origin": "AI_EDITOR",
        }
        
        if images:
            user_input_message["images"] = images
        
        # 只有在有 tools 或 tool_results 时才添加 userInputMessageContext
        context = {}
        if tools:
            context["tools"] = tools
        if tool_results:
            context["toolResults"] = tool_results
        
        if context:
            user_input_message["userInputMessageContext"] = context
        
        return {
            "conversationState": {
                "agentContinuationId": str(uuid.uuid4()),
                "agentTaskType": "vibe",
                "chatTriggerType": "MANUAL",
                "conversationId": conversation_id,
                "currentMessage": {"userInputMessage": user_input_message},
                "history": history or []
            }
        }
    
    # 上下文窗口大小（200k tokens）
    CONTEXT_WINDOW_SIZE = 200_000

    def parse_response(self, raw: bytes) -> Dict[str, Any]:
        """解析 AWS event-stream 格式响应"""
        result = {
            "content": [],
            "tool_uses": [],
            "tool_uses_raw": [],
            "stop_reason": "end_turn",
            "context_usage_percentage": None,  # 从 contextUsageEvent 获取
            "input_tokens": None,  # 从 contextUsageEvent 计算
            "stream_truncated": False,
            "tool_json_truncated": False,
            "only_thinking": False,  # 是否仅包含 thinking 内容（无 text/tool_use）
        }

        tool_input_buffer = {}
        pos = 0
        stream_truncated = False
        exception_stop_reason = None  # exception 事件设置的 stop_reason
        context_window_exceeded = False  # 上下文窗口是否用尽

        while pos < len(raw):
            if pos + 12 > len(raw):
                # 剩余字节不足以构成完整的 event-stream 消息头
                if result["content"] or tool_input_buffer:
                    stream_truncated = True
                break

            total_len = int.from_bytes(raw[pos:pos+4], 'big')
            headers_len = int.from_bytes(raw[pos+4:pos+8], 'big')

            if total_len == 0:
                break

            if total_len > len(raw) - pos:
                # 消息声明的长度超出可用数据，流被截断
                stream_truncated = True
                break

            header_start = pos + 12
            header_end = header_start + headers_len
            headers_data = raw[header_start:header_end]
            event_type = None

            try:
                headers_str = headers_data.decode('utf-8', errors='ignore')
                if 'toolUseEvent' in headers_str:
                    event_type = 'toolUseEvent'
                elif 'assistantResponseEvent' in headers_str:
                    event_type = 'assistantResponseEvent'
                elif 'contextUsageEvent' in headers_str:
                    event_type = 'contextUsageEvent'
                # AWS event-stream :message-type=exception 或 :exception-type header
                elif 'exception' in headers_str:
                    event_type = 'exception'
                    # 直接从 headers 中检测 ContentLengthExceededException
                    if 'ContentLengthExceeded' in headers_str:
                        exception_stop_reason = "max_tokens"
            except:
                pass

            payload_start = pos + 12 + headers_len
            payload_end = pos + total_len - 4

            if payload_start < payload_end:
                try:
                    payload = json.loads(raw[payload_start:payload_end].decode('utf-8'))

                    if _DEBUG_USAGE_ENABLED and isinstance(payload, dict):
                        logged_ctx_event = False
                        if event_type == 'contextUsageEvent' or 'contextUsageEvent' in payload:
                            ctx_event = payload.get('contextUsageEvent', payload)
                            _debug_usage_log("contextUsageEvent", ctx_event)
                            logged_ctx_event = True
                        usage_fields = _extract_usage_fields(payload)
                        if usage_fields and not logged_ctx_event:
                            _debug_usage_log("payload.usage_fields", usage_fields)
                        if "assistantResponseEvent" in payload:
                            assistant_usage = _extract_usage_fields(payload["assistantResponseEvent"])
                            if assistant_usage:
                                _debug_usage_log("assistantResponseEvent.usage_fields", assistant_usage)

                    if 'assistantResponseEvent' in payload:
                        e = payload['assistantResponseEvent']
                        if 'content' in e:
                            result["content"].append(e['content'])
                    elif 'content' in payload and event_type != 'toolUseEvent':
                        result["content"].append(payload['content'])

                    # 解析 contextUsageEvent
                    if event_type == 'contextUsageEvent' or 'contextUsageEvent' in payload:
                        ctx_event = payload.get('contextUsageEvent', payload)
                        if 'contextUsagePercentage' in ctx_event:
                            percentage = ctx_event['contextUsagePercentage']
                            result["context_usage_percentage"] = percentage
                            # 计算实际 input_tokens: percentage * 200000 / 100
                            result["input_tokens"] = int(percentage * self.CONTEXT_WINDOW_SIZE / 100)
                            # 上下文使用量达到 100% 时标记
                            if percentage >= 100:
                                context_window_exceeded = True

                    if event_type == 'toolUseEvent' or 'toolUseId' in payload:
                        tool_id = payload.get('toolUseId', '')
                        tool_name = payload.get('name', '')
                        tool_input = payload.get('input', '')

                        if tool_id:
                            if tool_id not in tool_input_buffer:
                                tool_input_buffer[tool_id] = {
                                    "id": tool_id,
                                    "name": tool_name,
                                    "input": ""
                                }
                            if tool_name and not tool_input_buffer[tool_id]["name"]:
                                tool_input_buffer[tool_id]["name"] = tool_name
                            if tool_input:
                                current_input = tool_input_buffer[tool_id]["input"]
                                if not current_input:
                                    tool_input_buffer[tool_id]["input"] = tool_input
                                elif tool_input.startswith(current_input):
                                    # Handle cumulative tool input by replacing with the newest payload.
                                    tool_input_buffer[tool_id]["input"] = tool_input
                                elif current_input.startswith(tool_input):
                                    # Ignore shorter duplicates.
                                    pass
                                else:
                                    tool_input_buffer[tool_id]["input"] = current_input + tool_input

                    # 解析 exception 事件（如 ContentLengthExceededException）
                    if event_type == 'exception':
                        exception_type = payload.get('__type', '') or payload.get('exceptionType', '')
                        if 'ContentLengthExceeded' in exception_type:
                            exception_stop_reason = "max_tokens"
                except:
                    pass

            pos += total_len

        # 检测工具调用 JSON 是否被截断（精细化检测，对应 kiro.rs handlers.rs:1051-1084）
        from ..core.truncation import detect_truncation, build_soft_failure_result

        tool_json_truncated = False
        for tool_id, tool_data in tool_input_buffer.items():
            input_str = tool_data["input"]
            try:
                json.loads(input_str)
            except (json.JSONDecodeError, ValueError):
                tool_json_truncated = True
                truncation_info = detect_truncation(
                    tool_data["name"], tool_data["id"], input_str
                )
                if truncation_info:
                    soft_msg = build_soft_failure_result(truncation_info)
                    print(
                        f"[Truncation] tool_use_id={tool_data['id']} "
                        f"type={truncation_info.truncation_type.value} "
                        f"检测到工具调用截断: {soft_msg}"
                    )

        # 组装工具调用（即使 JSON 不完整，也保留 raw）
        for tool_id, tool_data in tool_input_buffer.items():
            input_str = tool_data["input"]
            try:
                input_json = json.loads(input_str)
            except:
                input_json = {"raw": input_str}

            result["tool_uses"].append({
                "type": "tool_use",
                "id": tool_data["id"],
                "name": tool_data["name"],
                "input": input_json
            })
            result["tool_uses_raw"].append(input_str)

        # 决定 stop_reason（优先级：exception > tool_use > context_window_exceeded > only-thinking > end_turn）
        # 检测 only-thinking：有 thinking 内容但无 text 和 tool_use
        from ..converters import extract_thinking_from_content
        full_text = "".join(result["content"])
        thinking_text, remaining_text = extract_thinking_from_content(full_text)
        has_tool_use = bool(result["tool_uses"])
        is_only_thinking = bool(thinking_text) and (not remaining_text) and (not has_tool_use)
        result["only_thinking"] = is_only_thinking

        if exception_stop_reason:
            result["stop_reason"] = exception_stop_reason
        elif has_tool_use:
            result["stop_reason"] = "tool_use"
        elif context_window_exceeded:
            result["stop_reason"] = "model_context_window_exceeded"
        elif is_only_thinking:
            result["stop_reason"] = "max_tokens"
        # else: 保持默认 "end_turn"

        result["stream_truncated"] = stream_truncated
        result["tool_json_truncated"] = tool_json_truncated
        return result
    
    def parse_response_text(self, raw: bytes) -> str:
        """解析响应，只返回文本内容"""
        result = self.parse_response(raw)
        return "".join(result["content"]) or "[No response]"
    
    async def refresh_token(self) -> Tuple[bool, str]:
        """刷新 token"""
        if not self.credentials:
            return False, "无凭证信息"
        
        refresher = TokenRefresher(self.credentials)
        return await refresher.refresh()
    
    def is_quota_exceeded(self, status_code: int, error_text: str) -> bool:
        """检查是否为配额超限错误"""
        if status_code in {429, 503, 529}:
            return True
        
        keywords = ["rate limit", "quota", "too many requests", "throttl"]
        error_lower = error_text.lower()
        return any(kw in error_lower for kw in keywords)
