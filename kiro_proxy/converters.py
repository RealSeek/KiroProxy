"""协议转换模块 - Anthropic/OpenAI/Gemini <-> Kiro

增强版：参考 proxycast 实现
- 工具数量限制（最多 50 个）
- 工具描述截断（最多 500 字符）
- 历史消息交替修复
- OpenAI tool 角色消息处理
- tool_choice: required 支持
- web_search 特殊工具支持
- tool_results 去重
- 提示词注入（限制输出长度）
"""
import json
import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional

from .core.history_manager import estimate_tokens_from_text

# 常量
MAX_TOOL_DESCRIPTION_LENGTH = 500

# 追加到 Write 工具 description 末尾的内容
WRITE_TOOL_DESCRIPTION_SUFFIX = "- IMPORTANT: If the content to write exceeds 150 lines, you MUST only write the first 50 lines using this tool, then use `Edit` tool to append the remaining content in chunks of no more than 50 lines each. If needed, leave a unique placeholder to help append content. Do NOT attempt to write all content at once."

# 追加到 Edit 工具 description 末尾的内容
EDIT_TOOL_DESCRIPTION_SUFFIX = "- IMPORTANT: If the `new_string` content exceeds 50 lines, you MUST split it into multiple Edit calls, each replacing no more than 50 lines at a time. If used to append content, leave a unique placeholder to help append content. On the final chunk, do NOT include the placeholder."

# 追加到系统提示词的分块写入策略
SYSTEM_CHUNKED_POLICY = (
    "When the Write or Edit tool has content size limits, always comply silently. "
    "Never suggest bypassing these limits via alternative tools. "
    "Never ask the user whether to switch approaches. "
    "Complete all chunked operations without commentary."
)


def _estimate_output_tokens(result: dict) -> int:
    text = "".join(result.get("content", []))
    _, text_only = extract_thinking_from_content(text)
    output_tokens = estimate_tokens_from_text(text_only)
    tool_inputs_raw = result.get("tool_uses_raw") or []
    if tool_inputs_raw:
        for tool_input in tool_inputs_raw:
            output_tokens += estimate_tokens_from_text(tool_input)
    else:
        for tool_use in result.get("tool_uses", []):
            output_tokens += estimate_tokens_from_text(json.dumps(
                tool_use.get("input", {}),
                ensure_ascii=False,
                separators=(",", ":")
            ))
    return output_tokens


def estimate_output_tokens(result: dict) -> int:
    """估算输出 token 数量（文本 + 工具输入）"""
    return _estimate_output_tokens(result)


def get_prompt_injection() -> str:
    """获取提示词注入内容（如果启用）"""
    from .core import get_history_config
    config = get_history_config()
    if config.prompt_injection_enabled:
        return config.prompt_injection_content + "\n\n"
    return ""


def generate_session_id(messages: list) -> str:
    """基于消息内容生成会话ID"""
    content = json.dumps(messages[:3], sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def extract_images_from_content(content) -> Tuple[str, List[dict]]:
    """从消息内容中提取文本和图片
    
    Returns:
        (text_content, images_list)
    """
    if isinstance(content, str):
        return content, []
    
    if not isinstance(content, list):
        return str(content) if content else "", []
    
    text_parts = []
    images = []
    
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict):
            block_type = block.get("type", "")
            
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            
            elif block_type == "image":
                # Anthropic 格式
                source = block.get("source", {})
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                
                fmt = "jpeg"
                if "png" in media_type:
                    fmt = "png"
                elif "gif" in media_type:
                    fmt = "gif"
                elif "webp" in media_type:
                    fmt = "webp"
                
                if data:
                    images.append({
                        "format": fmt,
                        "source": {"bytes": data}
                    })
            
            elif block_type == "image_url":
                # OpenAI 格式
                image_url = block.get("image_url", {})
                url = image_url.get("url", "")
                
                if url.startswith("data:"):
                    match = re.match(r'data:image/(\w+);base64,(.+)', url)
                    if match:
                        fmt = match.group(1)
                        data = match.group(2)
                        images.append({
                            "format": fmt,
                            "source": {"bytes": data}
                        })
    
    return "\n".join(text_parts), images


def normalize_json_schema(schema: dict) -> dict:
    """规范化 JSON Schema，防止非法值导致上游 API 400 错误

    处理 MCP 工具定义中可能出现的非法值：
    - type: 必须是非空字符串，否则默认 "object"
    - properties: 必须是 dict，否则替换为 {}
    - required: 必须是字符串列表，过滤非字符串元素
    - additionalProperties: 仅允许 bool 或 dict
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    # type
    t = schema.get("type")
    if not isinstance(t, str) or not t:
        schema["type"] = "object"

    # properties
    if "properties" in schema and not isinstance(schema["properties"], dict):
        schema["properties"] = {}

    # required
    if "required" in schema:
        req = schema["required"]
        if not isinstance(req, list):
            schema["required"] = []
        else:
            schema["required"] = [r for r in req if isinstance(r, str)]

    # additionalProperties
    if "additionalProperties" in schema:
        ap = schema["additionalProperties"]
        if not isinstance(ap, (bool, dict)):
            schema["additionalProperties"] = True

    # 递归处理嵌套的 properties
    if isinstance(schema.get("properties"), dict):
        for key, value in schema["properties"].items():
            if isinstance(value, dict):
                schema["properties"][key] = normalize_json_schema(value)

    return schema


def truncate_description(desc: str, max_length: int = MAX_TOOL_DESCRIPTION_LENGTH) -> str:
    """截断工具描述"""
    if len(desc) <= max_length:
        return desc
    return desc[:max_length - 3] + "..."


# ==================== Anthropic 转换 ====================

def convert_anthropic_tools_to_kiro(tools: List[dict]) -> List[dict]:
    """将 Anthropic 工具格式转换为 Kiro 格式

    与 kiro.rs 保持一致：直接转换所有工具，不做过滤和数量限制。
    - 截断过长的描述（最多 10000 字符，与 kiro.rs 一致）
    """
    if not tools:
        return []

    kiro_tools = []

    for tool in tools:  # 不限制工具数量
        name = tool.get("name", "")
        description = tool.get("description", "")
        input_schema = tool.get("input_schema", {"type": "object", "properties": {}})

        # 对 Write/Edit 工具追加分块写入策略描述
        suffix = ""
        if name == "Write":
            suffix = WRITE_TOOL_DESCRIPTION_SUFFIX
        elif name == "Edit":
            suffix = EDIT_TOOL_DESCRIPTION_SUFFIX
        if suffix:
            description = description + "\n" + suffix

        # 截断描述（与 kiro.rs 一致，最多 10000 字符）
        if len(description) > 10000:
            description = description[:10000]

        kiro_tools.append({
            "toolSpecification": {
                "name": name,
                "description": description,
                "inputSchema": {
                    "json": normalize_json_schema(input_schema)
                }
            }
        })

    return kiro_tools


def fix_history_alternation(history: List[dict], model_id: str = "claude-sonnet-4") -> List[dict]:
    """修复历史记录，确保 user/assistant 严格交替，并验证 toolUses/toolResults 配对
    
    Kiro API 规则：
    1. 消息必须严格交替：user -> assistant -> user -> assistant
    2. 当 assistant 有 toolUses 时，下一条 user 必须有对应的 toolResults
    3. 当 assistant 没有 toolUses 时，下一条 user 不能有 toolResults
    """
    if not history:
        return history
    
    # 深拷贝以避免修改原始数据
    import copy
    history = copy.deepcopy(history)
    
    fixed = []
    
    for i, item in enumerate(history):
        is_user = "userInputMessage" in item
        is_assistant = "assistantResponseMessage" in item
        
        if is_user:
            # 检查上一条是否也是 user
            if fixed and "userInputMessage" in fixed[-1]:
                # 检查当前消息是否有 tool_results
                user_msg = item["userInputMessage"]
                ctx = user_msg.get("userInputMessageContext", {})
                has_tool_results = bool(ctx.get("toolResults"))
                
                if has_tool_results:
                    # 合并 tool_results 到上一条 user 消息
                    new_results = ctx["toolResults"]
                    last_user = fixed[-1]["userInputMessage"]
                    
                    if "userInputMessageContext" not in last_user:
                        last_user["userInputMessageContext"] = {}
                    
                    last_ctx = last_user["userInputMessageContext"]
                    if "toolResults" in last_ctx and last_ctx["toolResults"]:
                        last_ctx["toolResults"].extend(new_results)
                    else:
                        last_ctx["toolResults"] = new_results
                    continue
                else:
                    # 插入一个占位 assistant 消息（不带 toolUses）
                    fixed.append({
                        "assistantResponseMessage": {
                            "content": "I understand."
                        }
                    })
            
            # 验证 toolResults 与前一个 assistant 的 toolUses 配对
            if fixed and "assistantResponseMessage" in fixed[-1]:
                last_assistant = fixed[-1]["assistantResponseMessage"]
                has_tool_uses = bool(last_assistant.get("toolUses"))
                
                user_msg = item["userInputMessage"]
                ctx = user_msg.get("userInputMessageContext", {})
                has_tool_results = bool(ctx.get("toolResults"))
                
                if has_tool_uses and not has_tool_results:
                    # assistant 有 toolUses 但 user 没有 toolResults
                    # 这是不允许的，需要清除 assistant 的 toolUses
                    last_assistant.pop("toolUses", None)
                elif not has_tool_uses and has_tool_results:
                    # assistant 没有 toolUses 但 user 有 toolResults
                    # 这是不允许的，需要清除 user 的 toolResults
                    item["userInputMessage"].pop("userInputMessageContext", None)
                elif has_tool_uses and has_tool_results:
                    # 两者都有，需要精确按 ID 配对，移除孤立的 toolUse
                    tool_result_ids = set()
                    for tr in ctx.get("toolResults", []):
                        tr_id = tr.get("toolUseId")
                        if tr_id:
                            tool_result_ids.add(tr_id)

                    tool_uses = last_assistant.get("toolUses", [])
                    paired_tool_uses = [tu for tu in tool_uses if tu.get("toolUseId") in tool_result_ids]

                    if not paired_tool_uses:
                        # 所有 toolUse 都是孤立的，全部清除
                        last_assistant.pop("toolUses", None)
                        item["userInputMessage"].pop("userInputMessageContext", None)
                    elif len(paired_tool_uses) < len(tool_uses):
                        # 部分孤立，只保留有配对的
                        last_assistant["toolUses"] = paired_tool_uses
            
            fixed.append(item)
        
        elif is_assistant:
            # 检查上一条是否也是 assistant
            if fixed and "assistantResponseMessage" in fixed[-1]:
                # 插入一个占位 user 消息（不带 toolResults）
                fixed.append({
                    "userInputMessage": {
                        "content": "Continue",
                        "modelId": model_id,
                        "origin": "AI_EDITOR"
                    }
                })
            
            # 如果历史为空，先插入一个 user 消息
            if not fixed:
                fixed.append({
                    "userInputMessage": {
                        "content": "Continue",
                        "modelId": model_id,
                        "origin": "AI_EDITOR"
                    }
                })
            
            fixed.append(item)
    
    # 确保以 assistant 结尾（如果最后是 user，添加占位 assistant）
    if fixed and "userInputMessage" in fixed[-1]:
        # 不需要清除 toolResults，因为它是与前一个 assistant 的 toolUses 配对的
        # 占位 assistant 只是为了满足交替规则
        fixed.append({
            "assistantResponseMessage": {
                "content": "I understand."
            }
        })
    
    return fixed


def extract_thinking_from_content(text: str) -> Tuple[str, str]:
    """从响应文本中提取 <thinking> 标签内容

    Args:
        text: 包含可能的 <thinking> 标签的响应文本

    Returns:
        (thinking_text, remaining_text) - thinking 内容和剩余文本
    """
    if not text or "<thinking>" not in text:
        return "", text

    thinking_parts = []
    remaining = text

    # 提取所有完整的 <thinking>...</thinking> 块
    pattern = re.compile(r'<thinking>(.*?)</thinking>', re.DOTALL)
    matches = pattern.findall(remaining)
    if matches:
        # 剥离每段 thinking 内容开头的 \n（<thinking>\n 的 \n）
        thinking_parts.extend(m.lstrip('\n') for m in matches)
        remaining = pattern.sub('', remaining)

    # 处理未闭合的 <thinking> 标签（取到末尾）
    unclosed = re.search(r'<thinking>(.*)', remaining, re.DOTALL)
    if unclosed:
        # 同样剥离前导换行
        thinking_parts.append(unclosed.group(1).lstrip('\n'))
        remaining = remaining[:unclosed.start()]

    thinking_text = "\n".join(thinking_parts).strip()
    # 只去前导换行（</thinking>\n\n 后的 \n\n），保留其他空白
    remaining = remaining.lstrip('\n')
    return thinking_text, remaining


def generate_thinking_prefix(thinking: dict = None, output_config: dict = None) -> str:
    """生成 thinking 标签前缀

    Args:
        thinking: Anthropic thinking 配置，如 {"type": "enabled", "budget_tokens": 10000}
        output_config: Anthropic output_config 配置，如 {"effort": "high"}

    Returns:
        thinking 前缀字符串，如果不需要则返回空字符串
    """
    if not thinking:
        return ""

    thinking_type = thinking.get("type", "")
    if thinking_type == "enabled":
        budget_tokens = thinking.get("budget_tokens", 20000)
        return f"<thinking_mode>enabled</thinking_mode><max_thinking_length>{budget_tokens}</max_thinking_length>"
    elif thinking_type == "adaptive":
        effort = "high"
        if output_config and isinstance(output_config, dict):
            effort = output_config.get("effort", "high")
        return f"<thinking_mode>adaptive</thinking_mode><thinking_effort>{effort}</thinking_effort>"

    return ""


def convert_anthropic_messages_to_kiro(messages: List[dict], system="", thinking: dict = None, output_config: dict = None) -> Tuple[str, List[dict], List[dict]]:
    """将 Anthropic 消息格式转换为 Kiro 格式

    Args:
        messages: Anthropic 消息列表
        system: 系统消息
        thinking: Anthropic thinking 配置
        output_config: Anthropic output_config 配置

    Returns:
        (user_content, history, tool_results)
    """
    history = []
    user_content = ""
    current_tool_results = []

    # 处理 system
    system_text = ""
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                system_text += block.get("text", "") + "\n"
            elif isinstance(block, str):
                system_text += block + "\n"
        system_text = system_text.strip()
    elif isinstance(system, str):
        system_text = system
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        is_last = (i == len(messages) - 1)
        
        # 处理 content 列表
        tool_results = []
        text_parts = []
        
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        tr_content = block.get("content", "")
                        if isinstance(tr_content, list):
                            tr_text_parts = []
                            for tc in tr_content:
                                if isinstance(tc, dict) and tc.get("type") == "text":
                                    tr_text_parts.append(tc.get("text", ""))
                                elif isinstance(tc, str):
                                    tr_text_parts.append(tc)
                            tr_content = "\n".join(tr_text_parts)
                        
                        # 处理 is_error
                        status = "error" if block.get("is_error") else "success"
                        
                        tool_results.append({
                            "content": [{"text": str(tr_content)}],
                            "status": status,
                            "toolUseId": block.get("tool_use_id", "")
                        })
                elif isinstance(block, str):
                    text_parts.append(block)
            
            content = "\n".join(text_parts) if text_parts else ""
        
        # 处理工具结果
        if tool_results:
            # 去重
            seen_ids = set()
            unique_results = []
            for tr in tool_results:
                if tr["toolUseId"] not in seen_ids:
                    seen_ids.add(tr["toolUseId"])
                    unique_results.append(tr)
            tool_results = unique_results
            
            if is_last:
                current_tool_results = tool_results
                user_content = content if content else "Tool results provided."
            else:
                history.append({
                    "userInputMessage": {
                        "content": content if content else "Tool results provided.",
                        "modelId": "claude-sonnet-4",
                        "origin": "AI_EDITOR",
                        "userInputMessageContext": {
                            "toolResults": tool_results
                        }
                    }
                })
            continue
        
        if role == "user":
            # 在第一条用户消息时注入提示词和 system prompt
            if not history:
                injection = get_prompt_injection()
                # 生成 thinking 前缀
                thinking_prefix = generate_thinking_prefix(thinking, output_config)
                # 追加分块写入策略到系统消息
                full_system = f"{system_text}\n{SYSTEM_CHUNKED_POLICY}" if system_text else ""
                # 将 thinking 前缀放在最前面
                prefix = f"{thinking_prefix}{injection}" if thinking_prefix else injection
                if full_system:
                    content = f"{prefix}{full_system}\n\n{content}" if content else f"{prefix}{full_system}"
                elif prefix:
                    content = f"{prefix}{content}" if content else prefix.rstrip()

            if is_last:
                user_content = content if content else "Continue"
            else:
                history.append({
                    "userInputMessage": {
                        "content": content if content else "Continue",
                        "modelId": "claude-sonnet-4",
                        "origin": "AI_EDITOR"
                    }
                })
        
        elif role == "assistant":
            tool_uses = []
            assistant_text = ""
            thinking_text = ""

            if isinstance(msg.get("content"), list):
                text_parts = []
                for block in msg["content"]:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            tool_uses.append({
                                "toolUseId": block.get("id", ""),
                                "name": block.get("name", ""),
                                "input": block.get("input", {})
                            })
                        elif block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "thinking":
                            # 提取 thinking 块内容，用于续传
                            thinking_text += block.get("thinking", "")
                assistant_text = "\n".join(text_parts)
            else:
                assistant_text = content if isinstance(content, str) else ""

            # 组合 thinking 和 text 内容
            # 格式: <thinking>思考内容</thinking>\n\ntext内容
            # 这样模型在下一轮对话中能看到上次的思考内容，实现续传
            if thinking_text:
                if assistant_text and assistant_text.strip():
                    assistant_text = f"<thinking>{thinking_text}</thinking>\n\n{assistant_text}"
                else:
                    assistant_text = f"<thinking>{thinking_text}</thinking>"

            # 确保 assistant 消息有内容
            if not assistant_text:
                if tool_uses:
                    assistant_text = " "
                else:
                    assistant_text = "I understand."
            
            assistant_msg = {
                "assistantResponseMessage": {
                    "content": assistant_text
                }
            }
            # 只有在有 toolUses 时才添加这个字段
            if tool_uses:
                assistant_msg["assistantResponseMessage"]["toolUses"] = tool_uses
            
            history.append(assistant_msg)
    
    # 修复历史交替
    history = fix_history_alternation(history)
    
    return user_content, history, current_tool_results


def convert_kiro_response_to_anthropic(result: dict, model: str, msg_id: str) -> dict:
    """将 Kiro 响应转换为 Anthropic 格式"""
    content = []
    text = "".join(result["content"])

    # 提取 thinking 内容
    thinking_text, text = extract_thinking_from_content(text)
    if thinking_text:
        content.append({"type": "thinking", "thinking": thinking_text})

    if text:
        content.append({"type": "text", "text": text})
    elif thinking_text and not result.get("tool_uses"):
        # only-thinking 场景：补空 text 块确保 content 数组完整
        content.append({"type": "text", "text": " "})
    
    for tool_use in result["tool_uses"]:
        content.append(tool_use)
    
    input_tokens = result.get("input_tokens") or 0
    output_tokens = result.get("output_tokens")
    if output_tokens is None:
        output_tokens = _estimate_output_tokens(result)

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": result["stop_reason"],
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}
    }


# ==================== OpenAI 转换 ====================

def is_tool_choice_required(tool_choice) -> bool:
    """检查 tool_choice 是否为 required"""
    if isinstance(tool_choice, dict):
        t = tool_choice.get("type", "")
        return t in ("any", "tool", "required")
    elif isinstance(tool_choice, str):
        return tool_choice in ("required", "any")
    return False


def convert_openai_tools_to_kiro(tools: List[dict]) -> List[dict]:
    """将 OpenAI 工具格式转换为 Kiro 格式"""
    kiro_tools = []

    for tool in tools:
        tool_type = tool.get("type", "function")

        # 特殊工具
        if tool_type == "web_search":
            kiro_tools.append({
                "webSearchTool": {
                    "type": "web_search"
                }
            })
            continue

        if tool_type != "function":
            continue

        func = tool.get("function", {})
        name = func.get("name", "")
        description = func.get("description", f"Tool: {name}")

        # 对 Write/Edit 工具追加分块写入策略描述
        suffix = ""
        if name == "Write":
            suffix = WRITE_TOOL_DESCRIPTION_SUFFIX
        elif name == "Edit":
            suffix = EDIT_TOOL_DESCRIPTION_SUFFIX
        if suffix:
            description = description + "\n" + suffix

        # 截断描述（最多 10000 字符）
        if len(description) > 10000:
            description = description[:10000]
        parameters = func.get("parameters", {"type": "object", "properties": {}})

        kiro_tools.append({
            "toolSpecification": {
                "name": name,
                "description": description,
                "inputSchema": {
                    "json": normalize_json_schema(parameters)
                }
            }
        })

    return kiro_tools


def convert_openai_messages_to_kiro(
    messages: List[dict], 
    model: str,
    tools: List[dict] = None,
    tool_choice = None
) -> Tuple[str, List[dict], List[dict], List[dict]]:
    """将 OpenAI 消息格式转换为 Kiro 格式
    
    增强：
    - 支持 tool 角色消息
    - 支持 assistant 的 tool_calls
    - 支持 tool_choice: required
    - 历史交替修复
    
    Returns:
        (user_content, history, tool_results, kiro_tools)
    """
    system_content = ""
    history = []
    user_content = ""
    current_tool_results = []
    pending_tool_results = []  # 待处理的 tool 消息
    
    # 处理 tool_choice: required
    tool_instruction = ""
    if is_tool_choice_required(tool_choice) and tools:
        tool_instruction = "\n\n[CRITICAL INSTRUCTION] You MUST use one of the provided tools to respond. Do NOT respond with plain text. Call a tool function immediately."
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        is_last = (i == len(messages) - 1)
        
        # 提取文本内容
        if isinstance(content, list):
            content = " ".join([c.get("text", "") for c in content if c.get("type") == "text"])
        if not content:
            content = ""
        
        if role == "system":
            system_content = content + tool_instruction
        
        elif role == "tool":
            # OpenAI tool 角色消息 -> Kiro toolResults
            tool_call_id = msg.get("tool_call_id", "")
            pending_tool_results.append({
                "content": [{"text": str(content)}],
                "status": "success",
                "toolUseId": tool_call_id
            })
        
        elif role == "user":
            # 如果有待处理的 tool results，先处理
            if pending_tool_results:
                # 去重
                seen_ids = set()
                unique_results = []
                for tr in pending_tool_results:
                    if tr["toolUseId"] not in seen_ids:
                        seen_ids.add(tr["toolUseId"])
                        unique_results.append(tr)
                
                if is_last:
                    current_tool_results = unique_results
                else:
                    history.append({
                        "userInputMessage": {
                            "content": "Tool results provided.",
                            "modelId": model,
                            "origin": "AI_EDITOR",
                            "userInputMessageContext": {
                                "toolResults": unique_results
                            }
                        }
                    })
                pending_tool_results = []
            
            # 合并 system prompt 和提示词注入
            if not history:
                injection = get_prompt_injection()
                if system_content:
                    # 追加分块写入策略到系统消息
                    full_system = f"{system_content}\n{SYSTEM_CHUNKED_POLICY}"
                    content = f"{injection}{full_system}\n\n{content}"
                elif injection:
                    content = f"{injection}{content}" if content else injection.rstrip()
            
            if is_last:
                user_content = content
            else:
                history.append({
                    "userInputMessage": {
                        "content": content,
                        "modelId": model,
                        "origin": "AI_EDITOR"
                    }
                })
        
        elif role == "assistant":
            # 如果有待处理的 tool results，先创建 user 消息
            if pending_tool_results:
                seen_ids = set()
                unique_results = []
                for tr in pending_tool_results:
                    if tr["toolUseId"] not in seen_ids:
                        seen_ids.add(tr["toolUseId"])
                        unique_results.append(tr)
                
                history.append({
                    "userInputMessage": {
                        "content": "Tool results provided.",
                        "modelId": model,
                        "origin": "AI_EDITOR",
                        "userInputMessageContext": {
                            "toolResults": unique_results
                        }
                    }
                })
                pending_tool_results = []
            
            # 处理 tool_calls
            tool_uses = []
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except:
                    args = {}
                
                tool_uses.append({
                    "toolUseId": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "input": args
                })
            
            assistant_text = content if content else "I understand."
            
            assistant_msg = {
                "assistantResponseMessage": {
                    "content": assistant_text
                }
            }
            # 只有在有 toolUses 时才添加这个字段
            if tool_uses:
                assistant_msg["assistantResponseMessage"]["toolUses"] = tool_uses
            
            history.append(assistant_msg)
    
    # 处理末尾的 tool results
    if pending_tool_results:
        seen_ids = set()
        unique_results = []
        for tr in pending_tool_results:
            if tr["toolUseId"] not in seen_ids:
                seen_ids.add(tr["toolUseId"])
                unique_results.append(tr)
        current_tool_results = unique_results
        if not user_content:
            user_content = "Tool results provided."
    
    # 如果没有用户消息
    if not user_content:
        user_content = messages[-1].get("content", "") if messages else "Continue"
        if isinstance(user_content, list):
            user_content = " ".join([c.get("text", "") for c in user_content if c.get("type") == "text"])
        if not user_content:
            user_content = "Continue"
    
    # 历史不包含最后一条用户消息
    if history and "userInputMessage" in history[-1]:
        history = history[:-1]
    
    # 修复历史交替
    history = fix_history_alternation(history, model)
    
    # 转换工具
    kiro_tools = convert_openai_tools_to_kiro(tools) if tools else []
    
    return user_content, history, current_tool_results, kiro_tools


def convert_kiro_response_to_openai(result: dict, model: str, msg_id: str) -> dict:
    """将 Kiro 响应转换为 OpenAI 格式"""
    text = "".join(result["content"])
    tool_calls = []
    
    for tool_use in result.get("tool_uses", []):
        if tool_use.get("type") == "tool_use":
            tool_calls.append({
                "id": tool_use.get("id", ""),
                "type": "function",
                "function": {
                    "name": tool_use.get("name", ""),
                    "arguments": json.dumps(tool_use.get("input", {}))
                }
            })
    
    # 映射 stop_reason
    stop_reason = result.get("stop_reason", "stop")
    finish_reason = "tool_calls" if tool_calls else "stop"
    if stop_reason == "max_tokens":
        finish_reason = "length"
    
    message = {
        "role": "assistant",
        "content": text if text else None
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    input_tokens = result.get("input_tokens") or 0
    output_tokens = result.get("output_tokens")
    if output_tokens is None:
        output_tokens = _estimate_output_tokens(result)

    return {
        "id": msg_id,
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
    }


# ==================== Gemini 转换 ====================

def convert_gemini_tools_to_kiro(tools: List[dict]) -> List[dict]:
    """将 Gemini 工具格式转换为 Kiro 格式

    Gemini 工具格式：
    {
        "functionDeclarations": [
            {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {...}
            }
        ]
    }
    """
    kiro_tools = []

    for tool in tools:
        # Gemini 的工具定义在 functionDeclarations 中
        declarations = tool.get("functionDeclarations", [])

        for func in declarations:
            name = func.get("name", "")
            description = func.get("description", f"Tool: {name}")

            # 对 Write/Edit 工具追加分块写入策略描述
            suffix = ""
            if name == "Write":
                suffix = WRITE_TOOL_DESCRIPTION_SUFFIX
            elif name == "Edit":
                suffix = EDIT_TOOL_DESCRIPTION_SUFFIX
            if suffix:
                description = description + "\n" + suffix

            # 截断描述（最多 10000 字符）
            if len(description) > 10000:
                description = description[:10000]
            parameters = func.get("parameters", {"type": "object", "properties": {}})

            kiro_tools.append({
                "toolSpecification": {
                    "name": name,
                    "description": description,
                    "inputSchema": {
                        "json": normalize_json_schema(parameters)
                    }
                }
            })

    return kiro_tools


def convert_gemini_contents_to_kiro(
    contents: List[dict], 
    system_instruction: dict, 
    model: str,
    tools: List[dict] = None,
    tool_config: dict = None
) -> Tuple[str, List[dict], List[dict], List[dict]]:
    """将 Gemini 消息格式转换为 Kiro 格式
    
    增强：
    - 支持 functionCall 和 functionResponse
    - 支持 tool_config
    
    Returns:
        (user_content, history, tool_results, kiro_tools)
    """
    history = []
    user_content = ""
    current_tool_results = []
    pending_tool_results = []
    
    # 处理 system instruction
    system_text = ""
    if system_instruction:
        parts = system_instruction.get("parts", [])
        system_text = " ".join(p.get("text", "") for p in parts if "text" in p)
    
    # 处理 tool_config（类似 tool_choice）
    tool_instruction = ""
    if tool_config:
        mode = tool_config.get("functionCallingConfig", {}).get("mode", "")
        if mode in ("ANY", "REQUIRED"):
            tool_instruction = "\n\n[CRITICAL INSTRUCTION] You MUST use one of the provided tools to respond. Do NOT respond with plain text."
    
    for i, content in enumerate(contents):
        role = content.get("role", "user")
        parts = content.get("parts", [])
        is_last = (i == len(contents) - 1)
        
        # 提取文本和工具调用
        text_parts = []
        tool_calls = []
        tool_responses = []
        
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                # Gemini 的工具调用
                fc = part["functionCall"]
                tool_calls.append({
                    "toolUseId": fc.get("name", "") + "_" + str(i),  # Gemini 没有 ID，生成一个
                    "name": fc.get("name", ""),
                    "input": fc.get("args", {})
                })
            elif "functionResponse" in part:
                # Gemini 的工具响应
                fr = part["functionResponse"]
                response_content = fr.get("response", {})
                if isinstance(response_content, dict):
                    response_text = json.dumps(response_content)
                else:
                    response_text = str(response_content)
                
                tool_responses.append({
                    "content": [{"text": response_text}],
                    "status": "success",
                    "toolUseId": fr.get("name", "") + "_" + str(i - 1)  # 匹配上一个调用
                })
        
        text = " ".join(text_parts)
        
        if role == "user":
            # 处理待处理的 tool responses
            if pending_tool_results:
                seen_ids = set()
                unique_results = []
                for tr in pending_tool_results:
                    if tr["toolUseId"] not in seen_ids:
                        seen_ids.add(tr["toolUseId"])
                        unique_results.append(tr)
                
                history.append({
                    "userInputMessage": {
                        "content": "Tool results provided.",
                        "modelId": model,
                        "origin": "AI_EDITOR",
                        "userInputMessageContext": {
                            "toolResults": unique_results
                        }
                    }
                })
                pending_tool_results = []
            
            # 处理 functionResponse（用户消息中的工具响应）
            if tool_responses:
                pending_tool_results.extend(tool_responses)
            
            # 合并 system prompt 和提示词注入
            if not history:
                injection = get_prompt_injection()
                if system_text:
                    # 追加分块写入策略到系统消息
                    full_system = f"{system_text}\n{SYSTEM_CHUNKED_POLICY}"
                    text = f"{injection}{full_system}{tool_instruction}\n\n{text}"
                elif injection:
                    text = f"{injection}{text}" if text else injection.rstrip()
            
            if is_last:
                user_content = text
                if pending_tool_results:
                    current_tool_results = pending_tool_results
                    pending_tool_results = []
            else:
                if text:
                    history.append({
                        "userInputMessage": {
                            "content": text,
                            "modelId": model,
                            "origin": "AI_EDITOR"
                        }
                    })
        
        elif role == "model":
            # 处理待处理的 tool responses
            if pending_tool_results:
                seen_ids = set()
                unique_results = []
                for tr in pending_tool_results:
                    if tr["toolUseId"] not in seen_ids:
                        seen_ids.add(tr["toolUseId"])
                        unique_results.append(tr)
                
                history.append({
                    "userInputMessage": {
                        "content": "Tool results provided.",
                        "modelId": model,
                        "origin": "AI_EDITOR",
                        "userInputMessageContext": {
                            "toolResults": unique_results
                        }
                    }
                })
                pending_tool_results = []
            
            assistant_text = text if text else "I understand."
            
            assistant_msg = {
                "assistantResponseMessage": {
                    "content": assistant_text
                }
            }
            # 只有在有 toolUses 时才添加这个字段
            if tool_calls:
                assistant_msg["assistantResponseMessage"]["toolUses"] = tool_calls
            
            history.append(assistant_msg)
    
    # 处理末尾的 tool results
    if pending_tool_results:
        current_tool_results = pending_tool_results
        if not user_content:
            user_content = "Tool results provided."
    
    # 如果没有用户消息
    if not user_content:
        if contents:
            last_parts = contents[-1].get("parts", [])
            user_content = " ".join(p.get("text", "") for p in last_parts if "text" in p)
        if not user_content:
            user_content = "Continue"
    
    # 修复历史交替
    history = fix_history_alternation(history, model)
    
    # 移除最后一条（当前用户消息）
    if history and "userInputMessage" in history[-1]:
        history = history[:-1]
    
    # 转换工具
    kiro_tools = convert_gemini_tools_to_kiro(tools) if tools else []
    
    return user_content, history, current_tool_results, kiro_tools


def convert_kiro_response_to_gemini(result: dict, model: str) -> dict:
    """将 Kiro 响应转换为 Gemini 格式"""
    text = "".join(result.get("content", []))
    tool_uses = result.get("tool_uses", [])
    
    parts = []
    
    # 添加文本部分
    if text:
        parts.append({"text": text})
    
    # 添加工具调用
    for tool_use in tool_uses:
        if tool_use.get("type") == "tool_use":
            parts.append({
                "functionCall": {
                    "name": tool_use.get("name", ""),
                    "args": tool_use.get("input", {})
                }
            })
    
    # 映射 stop_reason
    stop_reason = result.get("stop_reason", "STOP")
    finish_reason = "STOP"
    if tool_uses:
        finish_reason = "TOOL_CALLS"
    elif stop_reason == "max_tokens":
        finish_reason = "MAX_TOKENS"

    input_tokens = result.get("input_tokens") or 0
    output_tokens = result.get("output_tokens")
    if output_tokens is None:
        output_tokens = _estimate_output_tokens(result)

    return {
        "candidates": [{
            "content": {
                "parts": parts,
                "role": "model"
            },
            "finishReason": finish_reason,
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": input_tokens,
            "candidatesTokenCount": output_tokens,
            "totalTokenCount": input_tokens + output_tokens
        }
    }
