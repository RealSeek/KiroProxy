"""Context compression pipeline - 上下文缩减管线

1:1 对应 kiro.rs src/anthropic/compressor.rs + handlers.rs 的逻辑：
- 5 层渐进式压缩（空白→thinking→tool_result→tool_use_input→历史）
- 工具配对修复（repair_tool_pairing_pass）
- 自适应请求体缩减循环（adaptive_shrink_request_body）
"""
import json
import re
from dataclasses import dataclass
from typing import Optional, Tuple


# ==================== 配置 ====================

@dataclass
class CompressionConfig:
    """压缩管线配置"""
    enabled: bool = True
    whitespace_compression: bool = True
    thinking_strategy: str = "discard"       # "discard" | "truncate" | "keep"
    tool_result_max_chars: int = 8000
    tool_result_head_lines: int = 80
    tool_result_tail_lines: int = 40
    tool_use_input_max_chars: int = 6000
    tool_description_max_chars: int = 4000
    max_history_turns: int = 80
    max_history_chars: int = 400_000
    max_request_body_bytes: int = 4_718_592  # ~4.5 MiB

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "whitespace_compression": self.whitespace_compression,
            "thinking_strategy": self.thinking_strategy,
            "tool_result_max_chars": self.tool_result_max_chars,
            "tool_result_head_lines": self.tool_result_head_lines,
            "tool_result_tail_lines": self.tool_result_tail_lines,
            "tool_use_input_max_chars": self.tool_use_input_max_chars,
            "tool_description_max_chars": self.tool_description_max_chars,
            "max_history_turns": self.max_history_turns,
            "max_history_chars": self.max_history_chars,
            "max_request_body_bytes": self.max_request_body_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CompressionConfig":
        return cls(
            enabled=data.get("enabled", True),
            whitespace_compression=data.get("whitespace_compression", True),
            thinking_strategy=data.get("thinking_strategy", "discard"),
            tool_result_max_chars=data.get("tool_result_max_chars", 8000),
            tool_result_head_lines=data.get("tool_result_head_lines", 80),
            tool_result_tail_lines=data.get("tool_result_tail_lines", 40),
            tool_use_input_max_chars=data.get("tool_use_input_max_chars", 6000),
            tool_description_max_chars=data.get("tool_description_max_chars", 4000),
            max_history_turns=data.get("max_history_turns", 80),
            max_history_chars=data.get("max_history_chars", 400_000),
            max_request_body_bytes=data.get("max_request_body_bytes", 4_718_592),
        )


@dataclass
class CompressionStats:
    """压缩统计"""
    whitespace_saved: int = 0
    thinking_saved: int = 0
    tool_result_saved: int = 0
    tool_use_input_saved: int = 0
    history_turns_removed: int = 0
    history_chars_removed: int = 0
    tool_uses_repaired: int = 0
    tool_results_repaired: int = 0


# ==================== 全局配置 ====================

_compression_config = CompressionConfig()


def get_compression_config() -> CompressionConfig:
    return _compression_config


def set_compression_config(config: CompressionConfig):
    global _compression_config
    _compression_config = config


def update_compression_config(data: dict):
    global _compression_config
    _compression_config = CompressionConfig.from_dict(data)


# ==================== Layer 1: 空白压缩 ====================

def compress_whitespace(text: str) -> str:
    """压缩空白：去尾部空白，折叠连续空行，去除前导空行（对应 compressor.rs:compress_whitespace）"""
    if text == " ":
        return text

    lines = text.split("\n")
    result = []
    consecutive_empty = 0

    for line in lines:
        stripped = line.rstrip()
        if stripped == "":
            consecutive_empty += 1
            # 与 Rust 一致：仅当 result 非空时才保留空行（去除前导空行）
            if consecutive_empty <= 2 and result:
                result.append("")
        else:
            consecutive_empty = 0
            result.append(stripped)

    return "\n".join(result)

def _compress_string_field(content: str) -> Tuple[str, int]:
    """压缩单个字符串字段，返回 (压缩后字符串, 节省字节数)
    对应 Rust compress_string_field：跳过 " " 占位符，返回字节差值"""
    if content == " ":
        return content, 0
    original_len = len(content.encode("utf-8"))
    compressed = compress_whitespace(content)
    compressed_len = len(compressed.encode("utf-8"))
    if compressed_len < original_len:
        return compressed, original_len - compressed_len
    return content, 0


def compress_whitespace_pass(state: dict) -> int:
    """Layer 1 入口，逐字段压缩并返回总节省字节数（对应 Rust compress_whitespace_pass）"""
    saved = 0

    for msg in state.get("history", []):
        if "userInputMessage" in msg:
            uim = msg["userInputMessage"]
            c = uim.get("content", "")
            if c:
                uim["content"], s = _compress_string_field(c)
                saved += s
        elif "assistantResponseMessage" in msg:
            arm = msg["assistantResponseMessage"]
            c = arm.get("content", "")
            if c:
                arm["content"], s = _compress_string_field(c)
                saved += s

    cm = state.get("currentMessage", {}).get("userInputMessage", {})
    c = cm.get("content", "")
    if c:
        cm["content"], s = _compress_string_field(c)
        saved += s

    return saved


# ==================== Layer 2: Thinking 块处理 ====================

_THINKING_PATTERN = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)
_THINKING_OPEN = "<thinking>"
_THINKING_CLOSE = "</thinking>"


def compress_thinking_pass(state: dict, strategy: str) -> int:
    """处理 assistant 消息中的 thinking 块（对应 compressor.rs:compress_thinking_pass）"""
    if strategy == "keep":
        return 0

    saved = 0
    for msg in state.get("history", []):
        if "assistantResponseMessage" not in msg:
            continue
        arm = msg["assistantResponseMessage"]
        content = arm.get("content", "")
        if not content or _THINKING_OPEN not in content:
            continue

        original_len = len(content)

        if strategy == "discard":
            # Remove complete <thinking>...</thinking> blocks
            content = _THINKING_PATTERN.sub("", content)
            # Handle unclosed thinking tag
            idx = content.find(_THINKING_OPEN)
            if idx != -1:
                content = content[:idx]
        elif strategy == "truncate":
            def _truncate_thinking(m):
                inner = m.group(0)[len(_THINKING_OPEN):-len(_THINKING_CLOSE)]
                if len(inner) > 500:
                    return f"{_THINKING_OPEN}{inner[:500]}...[truncated]{_THINKING_CLOSE}"
                return m.group(0)
            content = _THINKING_PATTERN.sub(_truncate_thinking, content)
            # Handle unclosed thinking tag
            idx = content.find(_THINKING_OPEN)
            if idx != -1 and _THINKING_CLOSE not in content[idx:]:
                inner = content[idx + len(_THINKING_OPEN):]
                if len(inner) > 500:
                    inner = inner[:500]
                content = content[:idx] + f"{_THINKING_OPEN}{inner}...[truncated]{_THINKING_CLOSE}"

        arm["content"] = content
        saved += max(0, original_len - len(content))

    return saved


# ==================== Layer 3: Tool Result 截断 ====================

def smart_truncate_by_lines(text: str, max_chars: int, head_lines: int, tail_lines: int) -> Tuple[str, int]:
    """智能按行截断（对应 compressor.rs:smart_truncate_by_lines）"""
    if len(text) <= max_chars:
        return text, 0

    # 使用 splitlines() 匹配 Rust text.lines()：不含尾部空行
    lines = text.splitlines()
    total_lines = len(lines)

    if total_lines <= head_lines + tail_lines:
        # 行数不够分，按字符对半截断（Rust 在此分支直接 return，不经过硬截断）
        half = max_chars // 2
        tail_chars = max_chars - half
        head_part = text[:half]
        tail_part = text[-tail_chars:] if tail_chars > 0 else ""
        omitted = len(text) - len(head_part) - len(tail_part)
        result = f"{head_part}\n... [{omitted} chars omitted] ...\n{tail_part}"
        saved = len(text) - len(result)
        return result, max(0, saved)
    else:
        head = "\n".join(lines[:head_lines])
        tail = "\n".join(lines[-tail_lines:])
        omitted_lines = total_lines - head_lines - tail_lines
        omitted_chars = len(text) - len(head) - len(tail)
        result = f"{head}\n... [{omitted_lines} lines omitted ({omitted_chars} chars)] ...\n{tail}"

    # 硬截断兜底：截断 result 本身（与 Rust safe_char_truncate(&result, max_chars) 一致）
    if len(result) > max_chars:
        result = result[:max_chars]

    saved = len(text) - len(result)
    return result, max(0, saved)


def compress_tool_results_pass(state: dict, max_chars: int, head_lines: int, tail_lines: int) -> int:
    """截断 toolResults 中的 text（对应 compressor.rs:compress_tool_results_pass）"""
    saved = 0

    def _process_tool_results(tool_results):
        nonlocal saved
        if not tool_results:
            return
        for tr in tool_results:
            for content_item in tr.get("content", []):
                text = content_item.get("text", "")
                if text and len(text) > max_chars:
                    truncated, s = smart_truncate_by_lines(text, max_chars, head_lines, tail_lines)
                    content_item["text"] = truncated
                    saved += s

    # history
    for msg in state.get("history", []):
        if "userInputMessage" in msg:
            ctx = msg["userInputMessage"].get("userInputMessageContext", {})
            _process_tool_results(ctx.get("toolResults"))

    # currentMessage
    cm = state.get("currentMessage", {}).get("userInputMessage", {})
    ctx = cm.get("userInputMessageContext", {})
    _process_tool_results(ctx.get("toolResults"))

    return saved


# ==================== Layer 4: Tool Use Input 截断 ====================

def truncate_json_value_strings(value, max_chars: int):
    """递归截断 JSON 值中的字符串（对应 compressor.rs:truncate_json_value_strings）"""
    if isinstance(value, str):
        if len(value) > max_chars:
            original_len = len(value)
            truncated = value[:max_chars]
            omitted_chars = original_len - max_chars
            with_marker = f"{truncated}...[truncated {omitted_chars} chars]"
            # 仅当带标记版本确实更短时才使用标记，否则退化为纯截断（与 Rust 一致）
            if len(with_marker) < original_len:
                return with_marker
            return truncated
        return value
    elif isinstance(value, dict):
        return {k: truncate_json_value_strings(v, max_chars) for k, v in value.items()}
    elif isinstance(value, list):
        return [truncate_json_value_strings(item, max_chars) for item in value]
    return value


def compress_tool_use_inputs_pass(state: dict, max_chars: int) -> int:
    """截断 assistant 消息中 toolUses 的 input（对应 compressor.rs:compress_tool_use_inputs_pass）"""
    saved = 0

    for msg in state.get("history", []):
        if "assistantResponseMessage" not in msg:
            continue
        arm = msg["assistantResponseMessage"]
        tool_uses = arm.get("toolUses")
        if not tool_uses:
            continue
        for tu in tool_uses:
            inp = tu.get("input")
            if inp is None:
                continue
            serialized = json.dumps(inp, ensure_ascii=False)
            if len(serialized) <= max_chars:
                continue
            original_len = len(serialized)
            tu["input"] = truncate_json_value_strings(inp, max_chars)
            new_len = len(json.dumps(tu["input"], ensure_ascii=False))
            saved += max(0, original_len - new_len)

    return saved


# ==================== Layer 5: 历史截断 ====================

def _message_chars(msg: dict) -> int:
    """计算单条消息的字符数（对应 Rust msg_bytes，仅计算 content）"""
    if "userInputMessage" in msg:
        return len(msg["userInputMessage"].get("content", ""))
    elif "assistantResponseMessage" in msg:
        return len(msg["assistantResponseMessage"].get("content", ""))
    return 0


def compress_history_pass(state: dict, max_turns: int, max_chars: int) -> Tuple[int, int]:
    """截断历史消息（对应 compressor.rs:compress_history_pass）

    保留前 2 条消息（系统对）+ 最新消息，从 index 2 开始成对删除。
    返回 (删除的轮数, 删除的字符数)
    """
    history = state.get("history", [])
    preserve_count = 2  # 保留前 2 条
    min_keep = preserve_count + 2  # 最少保留 4 条
    turns_removed = 0
    chars_removed = 0

    # 按轮数截断
    if max_turns > 0:
        max_messages = preserve_count + max_turns * 2
        while len(history) > max_messages and len(history) > min_keep:
            if preserve_count + 1 < len(history):
                chars_removed += _message_chars(history[preserve_count])
                chars_removed += _message_chars(history[preserve_count + 1]) if preserve_count + 1 < len(history) else 0
                del history[preserve_count:preserve_count + 2]
                turns_removed += 1
            else:
                break

    # 按字符数截断
    if max_chars > 0:
        while len(history) > min_keep:
            total = sum(_message_chars(m) for m in history)
            if total <= max_chars:
                break
            if preserve_count + 1 < len(history):
                chars_removed += _message_chars(history[preserve_count])
                chars_removed += _message_chars(history[preserve_count + 1]) if preserve_count + 1 < len(history) else 0
                del history[preserve_count:preserve_count + 2]
                turns_removed += 1
            else:
                break

    state["history"] = history
    return turns_removed, chars_removed


# ==================== Layer 6: 工具配对修复 ====================

def repair_tool_pairing_pass(state: dict) -> Tuple[int, int]:
    """修复 tool_use ↔ tool_result 配对（对应 compressor.rs:repair_tool_pairing_pass）

    返回 (移除的 tool_result 数, 移除的 tool_use 数)
    """
    history = state.get("history", [])
    removed_results = 0
    removed_uses = 0

    # Step 1: 收集所有 assistant toolUse IDs
    tool_use_ids = set()
    for msg in history:
        if "assistantResponseMessage" in msg:
            for tu in msg["assistantResponseMessage"].get("toolUses", []) or []:
                tid = tu.get("toolUseId")
                if tid:
                    tool_use_ids.add(tid)

    # Step 2: 移除 history + currentMessage 中孤立的 toolResults
    def _filter_tool_results(ctx):
        nonlocal removed_results
        trs = ctx.get("toolResults")
        if not trs:
            return
        filtered = [tr for tr in trs if tr.get("toolUseId") in tool_use_ids]
        removed_results += len(trs) - len(filtered)
        # 与 Rust retain 一致：保留列表（即使为空），不删除 key
        ctx["toolResults"] = filtered

    for msg in history:
        if "userInputMessage" in msg:
            ctx = msg["userInputMessage"].get("userInputMessageContext", {})
            _filter_tool_results(ctx)

    cm = state.get("currentMessage", {}).get("userInputMessage", {})
    ctx = cm.get("userInputMessageContext", {})
    _filter_tool_results(ctx)

    # Step 3: 收集所有 toolResult IDs
    tool_result_ids = set()
    for msg in history:
        if "userInputMessage" in msg:
            ctx = msg["userInputMessage"].get("userInputMessageContext", {})
            for tr in ctx.get("toolResults", []):
                tid = tr.get("toolUseId")
                if tid:
                    tool_result_ids.add(tid)
    cm_ctx = state.get("currentMessage", {}).get("userInputMessage", {}).get("userInputMessageContext", {})
    for tr in cm_ctx.get("toolResults", []):
        tid = tr.get("toolUseId")
        if tid:
            tool_result_ids.add(tid)

    # Step 4: 移除 assistant 中孤立的 toolUses
    for msg in history:
        if "assistantResponseMessage" not in msg:
            continue
        arm = msg["assistantResponseMessage"]
        tus = arm.get("toolUses")
        if not tus:
            continue
        filtered = [tu for tu in tus if tu.get("toolUseId") in tool_result_ids]
        removed_uses += len(tus) - len(filtered)
        if filtered:
            arm["toolUses"] = filtered
        else:
            arm.pop("toolUses", None)

    return removed_results, removed_uses


# ==================== 压缩入口 ====================

def compress(state: dict, config: CompressionConfig) -> CompressionStats:
    """执行 5 层渐进式压缩 + 工具配对修复（对应 compressor.rs:compress）"""
    stats = CompressionStats()

    if not config.enabled:
        return stats

    # Layer 1: 空白压缩
    if config.whitespace_compression:
        stats.whitespace_saved = compress_whitespace_pass(state)

    # Layer 2: Thinking 块处理
    if config.thinking_strategy != "keep":
        stats.thinking_saved = compress_thinking_pass(state, config.thinking_strategy)

    # Layer 3: Tool Result 截断
    if config.tool_result_max_chars > 0:
        stats.tool_result_saved = compress_tool_results_pass(
            state, config.tool_result_max_chars,
            config.tool_result_head_lines, config.tool_result_tail_lines
        )

    # Layer 4: Tool Use Input 截断
    if config.tool_use_input_max_chars > 0:
        stats.tool_use_input_saved = compress_tool_use_inputs_pass(
            state, config.tool_use_input_max_chars
        )

    # Layer 5: 历史截断
    if config.max_history_turns > 0 or config.max_history_chars > 0:
        turns, chars = compress_history_pass(
            state, config.max_history_turns, config.max_history_chars
        )
        stats.history_turns_removed = turns
        stats.history_chars_removed = chars

    # Layer 6: 工具配对修复（始终执行）
    results_repaired, uses_repaired = repair_tool_pairing_pass(state)
    stats.tool_results_repaired = results_repaired
    stats.tool_uses_repaired = uses_repaired

    return stats


# ==================== 长消息压缩（自适应缩减用） ====================

def compress_long_messages_pass(state: dict, max_chars: int) -> int:
    """截断过长的 user 消息 content（对应 compressor.rs:compress_long_messages_pass）"""
    if max_chars == 0:
        return 0

    saved = 0

    for msg in state.get("history", []):
        if "userInputMessage" not in msg:
            continue
        uim = msg["userInputMessage"]
        content = uim.get("content", "")
        if content == " " or len(content) <= max_chars:
            continue
        original_len = len(content)
        omitted = len(content) - max_chars
        new_content = content[:max_chars] + f"\n...[content truncated, {omitted} chars omitted]"
        uim["content"] = new_content
        saved += max(0, original_len - len(new_content))

    # currentMessage
    cm = state.get("currentMessage", {}).get("userInputMessage", {})
    content = cm.get("content", "")
    if content and content != " " and len(content) > max_chars:
        original_len = len(content)
        omitted = len(content) - max_chars
        new_content = content[:max_chars] + f"\n...[content truncated, {omitted} chars omitted]"
        cm["content"] = new_content
        saved += max(0, original_len - len(new_content))

    return saved


# ==================== 自适应缩减循环 ====================

# 常量（与 kiro.rs 完全一致）
ADAPTIVE_MAX_ITERS = 32
ADAPTIVE_MIN_TOOL_RESULT_MAX_CHARS = 512
ADAPTIVE_MIN_TOOL_USE_INPUT_MAX_CHARS = 256
ADAPTIVE_HISTORY_PRESERVE_MESSAGES = 2
ADAPTIVE_MIN_MESSAGE_CONTENT_MAX_CHARS = 8192


@dataclass
class AdaptiveOutcome:
    """自适应缩减结果"""
    iterations: int = 0
    initial_bytes: int = 0
    final_bytes: int = 0
    final_tool_result_max_chars: int = 0
    final_tool_use_input_max_chars: int = 0
    final_message_content_max_chars: int = 0
    history_messages_removed: int = 0


def _has_tool_results_or_tools(state: dict) -> bool:
    """检查 state 中是否有 toolResults 或 tools"""
    for msg in state.get("history", []):
        if "userInputMessage" in msg:
            ctx = msg["userInputMessage"].get("userInputMessageContext", {})
            if ctx.get("toolResults") or ctx.get("tools"):
                return True
    cm = state.get("currentMessage", {}).get("userInputMessage", {})
    ctx = cm.get("userInputMessageContext", {})
    return bool(ctx.get("toolResults") or ctx.get("tools"))


def _has_tool_uses(state: dict) -> bool:
    """检查 state 中是否有 toolUses"""
    for msg in state.get("history", []):
        if "assistantResponseMessage" in msg:
            if msg["assistantResponseMessage"].get("toolUses"):
                return True
    return False


def _max_user_message_chars(state: dict) -> int:
    """获取最大单条 user 消息的字符数（对应 Rust chars().count()，用于初始 message_content_max_chars）"""
    max_chars = 0
    for msg in state.get("history", []):
        if "userInputMessage" in msg:
            c = len(msg["userInputMessage"].get("content", ""))
            if c > max_chars:
                max_chars = c
    cm = state.get("currentMessage", {}).get("userInputMessage", {})
    c = len(cm.get("content", ""))
    if c > max_chars:
        max_chars = c
    return max_chars


def _max_user_message_bytes(state: dict) -> int:
    """获取最大单条 user 消息的字节数（对应 Rust content.len()，用于 L3 条件判断）"""
    max_bytes = 0
    for msg in state.get("history", []):
        if "userInputMessage" in msg:
            b = len(msg["userInputMessage"].get("content", "").encode("utf-8"))
            if b > max_bytes:
                max_bytes = b
    cm = state.get("currentMessage", {}).get("userInputMessage", {})
    b = len(cm.get("content", "").encode("utf-8"))
    if b > max_bytes:
        max_bytes = b
    return max_bytes


def adaptive_shrink_request_body(
    kiro_request: dict, base_config: CompressionConfig, max_body: int,
    skip_initial_compress: bool = False
) -> Tuple[str, Optional[AdaptiveOutcome]]:
    """自适应缩减请求体（对应 handlers.rs:adaptive_shrink_request_body）

    严格对齐 Rust 实现：4 层互斥（if/elif/else），每次迭代只执行一层。
    使用 adaptive_config 副本，循环末尾用 adaptive_config 重新压缩。
    """
    state = kiro_request.get("conversationState", {})

    # 初始压缩（可跳过，当 compress_and_prepare 已经做过时）
    if not skip_initial_compress:
        compress(state, base_config)
    body = json.dumps(kiro_request, ensure_ascii=False)
    body_bytes = len(body.encode("utf-8"))

    if max_body == 0 or body_bytes <= max_body or not base_config.enabled:
        return body, None

    outcome = AdaptiveOutcome(initial_bytes=body_bytes)

    # 创建 adaptive_config 副本（与 Rust adaptive_config = base_config.clone() 一致）
    adaptive_config = CompressionConfig.from_dict(base_config.to_dict())

    # 预扫描
    has_tr = _has_tool_results_or_tools(state)
    has_tu = _has_tool_uses(state)

    # 扫描最大 user content 字符数，计算初始 message_content_max_chars
    max_content_chars = _max_user_message_chars(state)
    message_content_max = max(max_content_chars * 3 // 4, ADAPTIVE_MIN_MESSAGE_CONTENT_MAX_CHARS)

    for iteration in range(ADAPTIVE_MAX_ITERS):
        if body_bytes <= max_body:
            break

        changed = False

        # 4 层互斥：if / elif / else { if / elif }（与 Rust 完全一致）
        if has_tr and adaptive_config.tool_result_max_chars > ADAPTIVE_MIN_TOOL_RESULT_MAX_CHARS:
            # L1: 缩减 tool_result_max_chars
            next_val = max(adaptive_config.tool_result_max_chars * 3 // 4, ADAPTIVE_MIN_TOOL_RESULT_MAX_CHARS)
            if next_val < adaptive_config.tool_result_max_chars:
                adaptive_config.tool_result_max_chars = next_val
                changed = True

        elif has_tu and adaptive_config.tool_use_input_max_chars > ADAPTIVE_MIN_TOOL_USE_INPUT_MAX_CHARS:
            # L2: 缩减 tool_use_input_max_chars
            next_val = max(adaptive_config.tool_use_input_max_chars * 3 // 4, ADAPTIVE_MIN_TOOL_USE_INPUT_MAX_CHARS)
            if next_val < adaptive_config.tool_use_input_max_chars:
                adaptive_config.tool_use_input_max_chars = next_val
                changed = True

        else:
            # L3 或 L4（互斥）
            max_single_bytes = _max_user_message_bytes(state)
            history = state.get("history", [])

            if ((max_single_bytes > max_body or len(history) <= ADAPTIVE_HISTORY_PRESERVE_MESSAGES + 2)
                    and message_content_max >= ADAPTIVE_MIN_MESSAGE_CONTENT_MAX_CHARS):
                # L3: 截断超长用户消息（先截断后递减，与 Rust 一致）
                saved = compress_long_messages_pass(state, message_content_max)
                if saved > 0:
                    changed = True
                outcome.final_message_content_max_chars = message_content_max
                message_content_max = max(message_content_max * 3 // 4, ADAPTIVE_MIN_MESSAGE_CONTENT_MAX_CHARS)

            elif len(history) > ADAPTIVE_HISTORY_PRESERVE_MESSAGES + 2:
                # L4: 移除最老历史消息（成对移除，单轮最多 16 条）
                preserve = ADAPTIVE_HISTORY_PRESERVE_MESSAGES
                min_len = preserve + 2
                removable = len(history) - min_len
                remove_msgs = min(removable, 16)
                remove_msgs -= remove_msgs % 2  # 保持成对
                if remove_msgs > 0:
                    del history[preserve:preserve + remove_msgs]
                    outcome.history_messages_removed += remove_msgs // 2
                    changed = True
                state["history"] = history

        if not changed:
            break

        # 用 adaptive_config 重新压缩 + 序列化（与 Rust 一致）
        compress(state, adaptive_config)
        body = json.dumps(kiro_request, ensure_ascii=False)
        body_bytes = len(body.encode("utf-8"))
        outcome.iterations = iteration + 1
        outcome.final_bytes = body_bytes

    if outcome.iterations == 0:
        outcome.final_bytes = body_bytes

    outcome.final_tool_result_max_chars = adaptive_config.tool_result_max_chars
    outcome.final_tool_use_input_max_chars = adaptive_config.tool_use_input_max_chars

    return body, outcome


# ==================== 封装入口 ====================

def compress_and_prepare(kiro_request: dict, config: CompressionConfig) -> str:
    """压缩 + 序列化，超限时自适应缩减。返回 JSON 字符串。

    Handler 调用此函数后，用 content= 发送预序列化的 body。
    """
    if not config.enabled:
        return json.dumps(kiro_request, ensure_ascii=False)

    state = kiro_request.get("conversationState", {})

    # 初始压缩（adaptive_shrink 内部不再重复调用）
    stats = compress(state, config)

    _log_compression_stats(stats)

    # 序列化
    body = json.dumps(kiro_request, ensure_ascii=False)
    body_bytes = len(body.encode("utf-8"))

    if body_bytes <= config.max_request_body_bytes:
        print(f"[Compressor] Body size: {body_bytes} bytes (within limit {config.max_request_body_bytes})")
        return body

    # 超限，启动自适应缩减（跳过初始压缩，因为上面已经做过了）
    print(f"[Compressor] Body size {body_bytes} exceeds limit {config.max_request_body_bytes}, starting adaptive shrink...")
    body, outcome = adaptive_shrink_request_body(kiro_request, config, config.max_request_body_bytes, skip_initial_compress=True)

    if outcome:
        print(
            f"[Compressor] Adaptive shrink: {outcome.iterations} iters, "
            f"{outcome.initial_bytes} -> {outcome.final_bytes} bytes, "
            f"tool_result_max={outcome.final_tool_result_max_chars}, "
            f"tool_use_input_max={outcome.final_tool_use_input_max_chars}, "
            f"msg_content_max={outcome.final_message_content_max_chars}, "
            f"history_removed={outcome.history_messages_removed}"
        )

    return body


def _log_compression_stats(stats: CompressionStats):
    """打印压缩统计日志"""
    parts = []
    if stats.whitespace_saved > 0:
        parts.append(f"whitespace={stats.whitespace_saved}")
    if stats.thinking_saved > 0:
        parts.append(f"thinking={stats.thinking_saved}")
    if stats.tool_result_saved > 0:
        parts.append(f"tool_result={stats.tool_result_saved}")
    if stats.tool_use_input_saved > 0:
        parts.append(f"tool_use_input={stats.tool_use_input_saved}")
    if stats.history_turns_removed > 0:
        parts.append(f"history_turns={stats.history_turns_removed}(-{stats.history_chars_removed}ch)")
    if stats.tool_results_repaired > 0 or stats.tool_uses_repaired > 0:
        parts.append(f"repair=tr:{stats.tool_results_repaired}/tu:{stats.tool_uses_repaired}")

    if parts:
        print(f"[Compressor] Saved: {', '.join(parts)}")

