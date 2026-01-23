"""WebSearch 处理器 - 通过 Kiro MCP 端点实现 Anthropic web_search 工具

参考: https://github.com/RealSeek/kiro2api/blob/main/server/websearch.go
"""
import json
import uuid
import time
import secrets
import httpx
from typing import List, Dict, Any, Optional, Tuple
from fastapi.responses import StreamingResponse

from ..config import MCP_API_URL
from ..kiro_api import build_headers


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
    """从消息中提取用户查询"""
    if not messages:
        return ""

    # 获取最后一条用户消息
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                return " ".join(text_parts)
    return ""


def generate_tool_use_id() -> str:
    """生成工具调用 ID，格式: toolu_{22hex}"""
    return f"toolu_{secrets.token_hex(11)}"


async def call_mcp_web_search(query: str, token: str, machine_id: str,
                               profile_arn: str = None, client_id: str = None) -> Tuple[bool, dict]:
    """调用 MCP API 执行 web_search

    Returns:
        (success, result_or_error)
    """
    # 构建 MCP 请求
    request_id = f"web_search_{secrets.token_hex(11)}_{int(time.time())}_{secrets.token_hex(4)}"

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

    headers = build_headers(token, machine_id, profile_arn, client_id)

    try:
        async with httpx.AsyncClient(verify=False, timeout=60) as client:
            resp = await client.post(MCP_API_URL, json=mcp_request, headers=headers)

            if resp.status_code != 200:
                return False, {"error": f"MCP API error: {resp.status_code}", "detail": resp.text[:500]}

            result = resp.json()

            # 检查 JSON-RPC 错误
            if "error" in result:
                return False, {"error": result["error"].get("message", "Unknown MCP error")}

            return True, result

    except Exception as e:
        return False, {"error": str(e)}


def parse_mcp_search_results(mcp_result: dict) -> List[dict]:
    """解析 MCP 搜索结果"""
    results = []

    # MCP 响应格式: {"result": {"content": [{"type": "text", "text": "..."}]}}
    content = mcp_result.get("result", {}).get("content", [])

    for item in content:
        if item.get("type") == "text":
            text = item.get("text", "")
            # 尝试解析为 JSON（搜索结果可能是 JSON 格式）
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    results.extend(parsed)
                elif isinstance(parsed, dict):
                    results.append(parsed)
                else:
                    results.append({"text": text})
            except json.JSONDecodeError:
                results.append({"text": text})

    return results


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
    profile_arn: str = None,
    client_id: str = None
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
        msg_id = f"msg_{uuid.uuid4().hex[:8]}"
        tool_use_id = generate_tool_use_id()

        # 1. message_start
        yield f'data: {{"type":"message_start","message":{{"id":"{msg_id}","type":"message","role":"assistant","content":[],"model":"{model}","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":100,"output_tokens":0}}}}}}\n\n'

        # 2. content_block_start - server_tool_use (web_search 调用)
        yield f'data: {{"type":"content_block_start","index":0,"content_block":{{"type":"server_tool_use","id":"{tool_use_id}","name":"web_search"}}}}\n\n'

        # 3. content_block_delta - input JSON
        input_json = json.dumps({"query": query})
        yield f'data: {{"type":"content_block_delta","index":0,"delta":{{"type":"input_json_delta","partial_json":{json.dumps(input_json)}}}}}\n\n'

        # 4. content_block_stop
        yield f'data: {{"type":"content_block_stop","index":0}}\n\n'

        # 5. 调用 MCP API 执行搜索
        success, result = await call_mcp_web_search(query, token, machine_id, profile_arn, client_id)

        if success:
            search_results = parse_mcp_search_results(result)
            results_text = format_search_results_text(search_results)

            # 6. content_block_start - web_search_tool_result
            yield f'data: {{"type":"content_block_start","index":1,"content_block":{{"type":"web_search_tool_result","tool_use_id":"{tool_use_id}","content":{json.dumps(search_results)}}}}}\n\n'
            yield f'data: {{"type":"content_block_stop","index":1}}\n\n'

            # 7. content_block_start - text (AI 总结)
            yield f'data: {{"type":"content_block_start","index":2,"content_block":{{"type":"text","text":""}}}}\n\n'

            # 8. 流式输出文本摘要
            summary = f"Based on the web search results for \"{query}\":\n\n{results_text}"

            # 分块输出
            chunk_size = 50
            for i in range(0, len(summary), chunk_size):
                chunk = summary[i:i+chunk_size]
                yield f'data: {{"type":"content_block_delta","index":2,"delta":{{"type":"text_delta","text":{json.dumps(chunk)}}}}}\n\n'

            yield f'data: {{"type":"content_block_stop","index":2}}\n\n'

        else:
            # 搜索失败，返回错误信息
            error_msg = result.get("error", "Web search failed")

            yield f'data: {{"type":"content_block_start","index":1,"content_block":{{"type":"text","text":""}}}}\n\n'
            yield f'data: {{"type":"content_block_delta","index":1,"delta":{{"type":"text_delta","text":{json.dumps(f"I apologize, but the web search encountered an error: {error_msg}. Let me try to help you with what I know.")}}}}}\n\n'
            yield f'data: {{"type":"content_block_stop","index":1}}\n\n'

        # 9. message_delta - stop_reason
        yield f'data: {{"type":"message_delta","delta":{{"stop_reason":"end_turn","stop_sequence":null}},"usage":{{"output_tokens":200}}}}\n\n'

        # 10. message_stop
        yield f'data: {{"type":"message_stop"}}\n\n'

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
