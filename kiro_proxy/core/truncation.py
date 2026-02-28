"""工具调用截断检测模块

当上游返回的工具调用 JSON 被截断时（例如因为 max_tokens 限制），
提供启发式检测和软失败恢复机制，引导模型分块重试。

对应 kiro.rs src/anthropic/truncation.rs
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TruncationType(Enum):
    """截断类型"""
    EMPTY_INPUT = "empty_input"
    INVALID_JSON = "invalid_json"
    MISSING_FIELDS = "missing_fields"
    INCOMPLETE_STRING = "incomplete_string"


@dataclass
class TruncationInfo:
    """截断检测结果"""
    truncation_type: TruncationType
    tool_name: str
    tool_use_id: str
    raw_input: str


def detect_truncation(
    tool_name: str,
    tool_use_id: str,
    raw_input: str,
) -> Optional[TruncationInfo]:
    """检测工具调用输入是否被截断

    启发式判断规则：
    1. 空输入 → EMPTY_INPUT
    2. 未闭合的引号 → INCOMPLETE_STRING
    3. 括号不平衡 → INVALID_JSON
    """
    trimmed = raw_input.strip()

    if not trimmed:
        return TruncationInfo(
            truncation_type=TruncationType.EMPTY_INPUT,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            raw_input=raw_input,
        )

    if _has_unclosed_string(trimmed):
        return TruncationInfo(
            truncation_type=TruncationType.INCOMPLETE_STRING,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            raw_input=raw_input,
        )

    if not _are_brackets_balanced(trimmed):
        return TruncationInfo(
            truncation_type=TruncationType.INVALID_JSON,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            raw_input=raw_input,
        )

    return None


def build_soft_failure_result(info: TruncationInfo) -> str:
    """构建软失败的工具结果消息

    当检测到截断时，生成一条引导模型分块重试的错误消息，
    而不是直接返回解析错误。
    """
    if info.truncation_type == TruncationType.EMPTY_INPUT:
        return (
            f"Tool call '{info.tool_name}' (id: {info.tool_use_id}) was truncated: "
            f"the input was empty. This usually means the response was cut off due to "
            f"token limits. Please retry with a shorter input or break the operation "
            f"into smaller steps."
        )
    elif info.truncation_type == TruncationType.INCOMPLETE_STRING:
        return (
            f"Tool call '{info.tool_name}' (id: {info.tool_use_id}) was truncated: "
            f"a string value was not properly closed. The input appears to have been "
            f"cut off mid-string. Please retry with shorter content or split the "
            f"operation into multiple calls."
        )
    elif info.truncation_type == TruncationType.INVALID_JSON:
        return (
            f"Tool call '{info.tool_name}' (id: {info.tool_use_id}) was truncated: "
            f"the JSON input is incomplete (unbalanced brackets). Please retry with "
            f"a shorter input or break the operation into smaller steps."
        )
    elif info.truncation_type == TruncationType.MISSING_FIELDS:
        return (
            f"Tool call '{info.tool_name}' (id: {info.tool_use_id}) was truncated: "
            f"required fields are missing. Please retry with all required fields included."
        )
    return ""


def _has_unclosed_string(s: str) -> bool:
    """检查字符串中是否有未闭合的引号"""
    in_string = False
    escape_next = False

    for ch in s:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
        elif ch == '"':
            in_string = not in_string

    return in_string


def _are_brackets_balanced(s: str) -> bool:
    """检查括号是否平衡"""
    stack = []
    in_string = False
    escape_next = False

    for ch in s:
        if escape_next:
            escape_next = False
            continue
        if in_string:
            if ch == '\\':
                escape_next = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in ('{', '['):
            stack.append(ch)
        elif ch == '}':
            if not stack or stack[-1] != '{':
                return False
            stack.pop()
        elif ch == ']':
            if not stack or stack[-1] != '[':
                return False
            stack.pop()

    return len(stack) == 0
