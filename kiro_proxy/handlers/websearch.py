"""WebSearch 处理器 - 通过 Kiro MCP 端点实现 Anthropic web_search 工具

参考: https://github.com/RealSeek/kiro2api/blob/main/server/websearch.go
"""
import json
import uuid
import time
import secrets
import string
import httpx
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from fastapi.responses import StreamingResponse

from ..config import MCP_API_URL_TEMPLATE
from ..credential import generate_machine_id, get_kiro_version, get_system_info
from ..core.history_manager import estimate_tokens_from_text


def has_web_search_tool(tools: List[dict]) -> bool:
    """检查请求中是否包含 web_search 工具"""
    if not tools:
        return False
    for tool in tools:
        tool_type = tool.get("type", "")
        name = tool.get("name", "")
        if "web_search" in tool_type or name in ("web_search", "web_search_20250305"):
            return True
    return False


def get_web_search_tool(tools: List[dict]) -> Optional[dict]:
    """获取 web_search 工具配置"""
    if not tools:
        return None
    for tool in tools:
        tool_type = tool.get("type", "")
        name = tool.get("name", "")
        if "web_search" in tool_type or name in ("web_search", "web_search_20250305"):
            return tool
    return None


def extract_user_query(messages: List[dict]) -> str:
    """从消息中提取用户查询

    读取第一条消息的第一个文本内容块，并去除 "Perform a web search for the query: " 前缀。
    与 kiro.rs 的 extract_search_query 保持一致。
    """
    if not messages:
        return ""

    # 获取第一条消息（与 kiro.rs 一致）
    first_msg = messages[0]
    content = first_msg.get("content", "")

    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        # 获取第一个 text 类型的内容块
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                break
            elif isinstance(block, str):
                text = block
                break

    # 去除 "Perform a web search for the query: " 前缀
    prefix = "Perform a web search for the query: "
    if text.startswith(prefix):
        text = text[len(prefix):]

    return text


def _extract_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            parts.append(_extract_text(item))
        return "".join(parts)
    if isinstance(content, dict):
        if "text" in content and isinstance(content.get("text"), str):
            return content["text"]
        if "content" in content:
            return _extract_text(content.get("content"))
    return str(content) if content else ""


def _estimate_input_tokens(messages: List[dict], system: Any = None) -> int:
    text = _extract_text(system)
    for msg in messages or []:
        text += _extract_text(msg.get("content"))
    return estimate_tokens_from_text(text)


def generate_tool_use_id() -> str:
    """生成服务端工具调用 ID，格式: srvtoolu_{32hex}"""
    return f"srvtoolu_{uuid.uuid4().hex}"


def _generate_random_alnum(length: int) -> str:
    """生成指定长度的大小写字母+数字随机字符串"""
    charset = string.ascii_letters + string.digits
    return "".join(secrets.choice(charset) for _ in range(length))


def _generate_random_lower_alnum(length: int) -> str:
    """生成指定长度的小写字母+数字随机字符串"""
    charset = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(charset) for _ in range(length))


def build_mcp_headers(token: str, machine_id: str) -> dict:
    """构建 MCP API 专用请求头（不含 kiro-agent-mode / codewhisperer-optout）"""
    kiro_version = get_kiro_version()
    os_name, node_version = get_system_info()

    return {
        "content-type": "application/json",
        "x-amz-user-agent": f"aws-sdk-js/1.0.27 KiroIDE-{kiro_version}-{machine_id}",
        "user-agent": (
            f"aws-sdk-js/1.0.27 ua/2.1 os/{os_name} lang/js "
            f"md/nodejs#{node_version} api/codewhispererstreaming#1.0.27 "
            f"m/E KiroIDE-{kiro_version}-{machine_id}"
        ),
        "amz-sdk-invocation-id": str(uuid.uuid4()),
        "amz-sdk-request": "attempt=1; max=3",
        "Authorization": f"Bearer {token}",
        "Connection": "close",
    }


async def call_mcp_web_search(
    query: str,
    token: str,
    machine_id: str,
    region: str = "us-east-1",
) -> Tuple[bool, dict]:
    """调用 MCP API 执行 web_search

    Returns:
        (success, result_or_error)
    """
    # 构建 MCP 请求（ID 格式与 kiro.rs 一致: web_search_tooluse_{22}_{ts}_{8}）
    random_22 = _generate_random_alnum(22)
    timestamp = int(time.time() * 1000)
    random_8 = _generate_random_lower_alnum(8)
    request_id = f"web_search_tooluse_{random_22}_{timestamp}_{random_8}"

    mcp_request = {
        "id": request_id,
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "web_search",
            "arguments": {
                "query": query
            }
        }
    }

    # MCP 专用 headers（不含 agent-mode / optout，含 host）
    headers = build_mcp_headers(token, machine_id)
    mcp_domain = f"q.{region}.amazonaws.com"
    headers["host"] = mcp_domain

    mcp_url = MCP_API_URL_TEMPLATE.format(region=region)

    print(f"[WebSearch] MCP URL: {mcp_url}")
    print(f"[WebSearch] Query: {query}")
    print(f"[WebSearch] Request ID: {request_id}")

    try:
        async with httpx.AsyncClient(verify=False, timeout=60) as client:
            resp = await client.post(mcp_url, json=mcp_request, headers=headers)

            print(f"[WebSearch] MCP response status: {resp.status_code}")

            if resp.status_code != 200:
                detail = resp.text[:500]
                print(f"[WebSearch] MCP error response: {detail}")
                return False, {"error": f"MCP API error: {resp.status_code}", "detail": detail}

            result = resp.json()

            # 检查 JSON-RPC 错误（error 为 null 时跳过）
            if result.get("error"):
                err_msg = result["error"].get("message", "Unknown MCP error")
                print(f"[WebSearch] MCP JSON-RPC error: {err_msg}")
                return False, {"error": err_msg}

            print(f"[WebSearch] MCP success, result keys: {list(result.keys())}")
            return True, result

    except Exception as e:
        print(f"[WebSearch] MCP exception: {e}")
        return False, {"error": str(e)}


def parse_mcp_search_results(mcp_result: dict) -> List[dict]:
    """解析 MCP 搜索结果

    MCP 响应格式: {"result": {"content": [{"type": "text", "text": "..."}]}}
    其中 text 是 JSON 字符串，格式为 {"results": [...], "totalResults": N}
    """
    results = []

    content = mcp_result.get("result", {}).get("content", [])

    for item in content:
        if item.get("type") == "text":
            text = item.get("text", "")
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "results" in parsed:
                    # WebSearchResults 包装结构: {"results": [...], "totalResults": N}
                    results.extend(parsed["results"])
                elif isinstance(parsed, list):
                    results.extend(parsed)
                elif isinstance(parsed, dict):
                    results.append(parsed)
                else:
                    results.append({"text": text})
            except json.JSONDecodeError:
                results.append({"text": text})

    return results


def _convert_page_age(published_date) -> Optional[str]:
    """将毫秒时间戳转换为人类可读的日期字符串（如 "February 28, 2026"）"""
    if published_date is None:
        return None
    try:
        ms = int(published_date)
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        # %B = full month, %d = day (strip leading zero), %Y = year
        return dt.strftime("%B %d, %Y").replace(" 0", " ")
    except (ValueError, TypeError, OSError):
        return None


def transform_to_web_search_results(results: List[dict]) -> List[dict]:
    """将 MCP 搜索结果转换为 Anthropic web_search_result 格式"""
    transformed = []
    for r in results:
        if not isinstance(r, dict):
            continue
        page_age = _convert_page_age(
            r.get("publishedDate") or r.get("published_date")
        )
        item = {
            "type": "web_search_result",
            "title": r.get("title", r.get("name", "")),
            "url": r.get("url", r.get("link", "")),
            "encrypted_content": r.get("snippet", r.get("description", r.get("text", ""))),
            "page_age": page_age,
        }
        transformed.append(item)
    return transformed


def format_search_results_text(results: List[dict]) -> str:
    """将搜索结果格式化为文本摘要"""
    if not results:
        return "No search results found."

    lines = ["Here are the web search results:\n"]

    for i, result in enumerate(results[:10], 1):  # 最多显示 10 条
        if isinstance(result, dict):
            title = result.get("title", result.get("name", ""))
            url = result.get("url", result.get("link", ""))
            snippet = result.get("snippet", result.get("description", result.get("text", "")))

            if title or url:
                lines.append(f"{i}. {title}")
                if url:
                    lines.append(f"   URL: {url}")
                if snippet:
                    lines.append(f"   {snippet[:200]}...")
                lines.append("")
        elif isinstance(result, str):
            lines.append(f"{i}. {result[:300]}")
            lines.append("")

    return "\n".join(lines)


async def handle_web_search_request(
    body: dict,
    token: str,
    machine_id: str,
    model: str,
    region: str = "us-east-1",
) -> StreamingResponse:
    """处理包含 web_search 的请求，返回 Anthropic 格式的流式响应"""

    messages = body.get("messages", [])
    tools = body.get("tools", [])

    # 提取用户查询
    query = extract_user_query(messages)
    if not query:
        query = "latest news"  # 默认查询

    # 获取 web_search 工具配置
    ws_tool = get_web_search_tool(tools)
    max_uses = ws_tool.get("max_uses", 5) if ws_tool else 5

    async def generate():
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        tool_use_id = generate_tool_use_id()
        input_tokens = _estimate_input_tokens(messages, body.get("system"))

        # 1. message_start
        yield f'event: message_start\ndata: {{"type":"message_start","message":{{"id":"{msg_id}","type":"message","role":"assistant","content":[],"model":"{model}","stop_reason":null,"usage":{{"input_tokens":{input_tokens},"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}}}}\n\n'

        # 2. content_block_start - text (搜索决策说明, index 0)
        decision_text = f'I\'ll search for "{query}".'
        yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":0,"content_block":{{"type":"text","text":""}}}}\n\n'
        yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":{json.dumps(decision_text)}}}}}\n\n'
        yield f'event: content_block_stop\ndata: {{"type":"content_block_stop","index":0}}\n\n'

        # 3. content_block_start - server_tool_use (web_search 调用, index 1)
        # server_tool_use 是服务端工具，input 在 content_block_start 中一次性完整发送，
        # 不像客户端 tool_use 需要通过 input_json_delta 增量传输。
        input_json = json.dumps({"query": query})
        yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":1,"content_block":{{"type":"server_tool_use","id":"{tool_use_id}","name":"web_search","input":{{"query":{json.dumps(query)}}}}}}}\n\n'

        # 4. content_block_stop (server_tool_use)
        yield f'event: content_block_stop\ndata: {{"type":"content_block_stop","index":1}}\n\n'

        # 调用 MCP API 执行搜索
        success, result = await call_mcp_web_search(query, token, machine_id, region=region)

        if success:
            search_results = parse_mcp_search_results(result)
            web_results = transform_to_web_search_results(search_results)
            results_text = format_search_results_text(search_results)

            # 5. content_block_start - web_search_tool_result (index 2)
            # 官方 API 的 web_search_tool_result 没有 tool_use_id 字段
            yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":2,"content_block":{{"type":"web_search_tool_result","content":{json.dumps(web_results)}}}}}\n\n'
            # 6. content_block_stop (web_search_tool_result)
            yield f'event: content_block_stop\ndata: {{"type":"content_block_stop","index":2}}\n\n'

            # 7. content_block_start - text (AI 总结, index 3)
            yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":3,"content_block":{{"type":"text","text":""}}}}\n\n'

            # 8. content_block_delta (text_delta) - 流式输出文本摘要
            summary = f"Based on the web search results for \"{query}\":\n\n{results_text}"
            output_tokens = estimate_tokens_from_text(summary) + estimate_tokens_from_text(input_json)

            # 分块输出
            chunk_size = 50
            for i in range(0, len(summary), chunk_size):
                chunk = summary[i:i+chunk_size]
                yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":3,"delta":{{"type":"text_delta","text":{json.dumps(chunk)}}}}}\n\n'

            # 9. content_block_stop (text)
            yield f'event: content_block_stop\ndata: {{"type":"content_block_stop","index":3}}\n\n'

        else:
            # 搜索失败，返回错误信息
            error_msg = result.get("error", "Web search failed")
            output_tokens = estimate_tokens_from_text(str(error_msg)) + estimate_tokens_from_text(input_json)

            yield f'event: content_block_start\ndata: {{"type":"content_block_start","index":2,"content_block":{{"type":"text","text":""}}}}\n\n'
            yield f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":2,"delta":{{"type":"text_delta","text":{json.dumps(f"I apologize, but the web search encountered an error: {error_msg}. Let me try to help you with what I know.")}}}}}\n\n'
            yield f'event: content_block_stop\ndata: {{"type":"content_block_stop","index":2}}\n\n'

        # 10. message_delta - stop_reason (含 server_tool_use.web_search_requests)
        # 官方 API 的 message_delta.delta 中没有 stop_sequence 字段
        yield f'event: message_delta\ndata: {{"type":"message_delta","delta":{{"stop_reason":"end_turn"}},"usage":{{"output_tokens":{output_tokens},"server_tool_use":{{"web_search_requests":1}}}}}}\n\n'

        # 11. message_stop
        yield f'event: message_stop\ndata: {{"type":"message_stop"}}\n\n'

    return StreamingResponse(generate(), media_type="text/event-stream")


def filter_web_search_tools(tools: List[dict]) -> List[dict]:
    """过滤掉 web_search 工具，返回其他工具"""
    if not tools:
        return []

    filtered = []
    for tool in tools:
        tool_type = tool.get("type", "")
        name = tool.get("name", "")
        if "web_search" not in tool_type and name not in ("web_search", "web_search_20250305"):
            filtered.append(tool)

    return filtered
