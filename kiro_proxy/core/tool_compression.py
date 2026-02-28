"""工具定义压缩模块

1:1 对应 kiro.rs src/anthropic/tool_compression.rs：
- 当工具定义的总序列化大小超过阈值时，通过两步压缩减小体积：
  1. 简化 input_schema：移除非必要字段（description 等），仅保留结构骨架
  2. 按比例截断 description：根据超出比例缩短描述，最短保留 50 字符
"""
import json
import copy
from typing import List

# 工具定义总大小阈值（20KB）
TOOL_SIZE_THRESHOLD = 20 * 1024

# description 最短保留字符数
MIN_DESCRIPTION_CHARS = 50


def compress_tools_if_needed(tools: List[dict]) -> List[dict]:
    """如果工具定义总大小超过阈值，执行压缩

    返回压缩后的工具列表（如果未超阈值则原样返回）。
    """
    total_size = _estimate_tools_size(tools)
    if total_size <= TOOL_SIZE_THRESHOLD:
        return tools

    print(
        f"[ToolCompression] Tool definitions ({total_size} bytes, "
        f"{len(tools)} tools) exceed threshold {TOOL_SIZE_THRESHOLD}, compressing..."
    )

    # 第一步：简化 input_schema
    compressed = [_simplify_tool(t) for t in tools]

    size_after_schema = _estimate_tools_size(compressed)
    if size_after_schema <= TOOL_SIZE_THRESHOLD:
        print(
            f"[ToolCompression] Schema simplification sufficient: "
            f"{total_size} -> {size_after_schema} bytes"
        )
        return compressed

    # 第二步：按比例截断 description（基于字节大小，与 kiro.rs 对齐）
    ratio = TOOL_SIZE_THRESHOLD / size_after_schema
    for tool in compressed:
        spec = tool.get("toolSpecification")
        if not spec:
            continue
        desc = spec.get("description", "")
        desc_bytes = len(desc.encode("utf-8"))
        target_bytes = int(desc_bytes * ratio)
        # 最短保留 MIN_DESCRIPTION_CHARS 个字符对应的字节数
        min_bytes = len(desc[:MIN_DESCRIPTION_CHARS].encode("utf-8")) if len(desc) >= MIN_DESCRIPTION_CHARS else desc_bytes
        target_bytes = max(target_bytes, min_bytes)
        if desc_bytes > target_bytes:
            # UTF-8 安全截断：逐字符累加字节数，找到不超过 target_bytes 的最大字符边界
            byte_count = 0
            truncate_at = 0
            for i, ch in enumerate(desc):
                ch_bytes = len(ch.encode("utf-8"))
                if byte_count + ch_bytes > target_bytes:
                    break
                byte_count += ch_bytes
                truncate_at = i + 1
            spec["description"] = desc[:truncate_at]

    final_size = _estimate_tools_size(compressed)
    print(
        f"[ToolCompression] Done: {total_size} -> {size_after_schema} (schema) "
        f"-> {final_size} (desc truncation)"
    )

    return compressed


def _estimate_tools_size(tools: List[dict]) -> int:
    """估算工具列表的总序列化大小（字节）

    与 kiro.rs estimate_tools_size 对齐：使用 UTF-8 字节数。
    """
    total = 0
    for tool in tools:
        spec = tool.get("toolSpecification")
        if not spec:
            # webSearchTool 等非标准工具，按 JSON 序列化估算
            total += len(json.dumps(tool, ensure_ascii=False).encode("utf-8"))
            continue
        total += len(spec.get("name", "").encode("utf-8"))
        total += len(spec.get("description", "").encode("utf-8"))
        schema = spec.get("inputSchema", {}).get("json")
        if schema is not None:
            total += len(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
    return total


def _simplify_tool(tool: dict) -> dict:
    """简化单个工具的 input_schema"""
    spec = tool.get("toolSpecification")
    if not spec:
        return copy.deepcopy(tool)

    schema = spec.get("inputSchema", {}).get("json")
    simplified_schema = _simplify_json_schema(schema) if schema else schema

    return {
        "toolSpecification": {
            "name": spec.get("name", ""),
            "description": spec.get("description", ""),
            "inputSchema": {
                "json": simplified_schema
            }
        }
    }


def _simplify_json_schema(schema) -> dict:
    """递归简化 JSON Schema

    保留结构骨架（type, properties 的 key 和 type, required），
    移除 properties 内部的 description、examples 等非必要字段。
    """
    if not isinstance(schema, dict):
        return schema

    result = {}

    # 保留顶层结构字段
    for key in ("$schema", "type", "required", "additionalProperties"):
        if key in schema:
            result[key] = schema[key]

    # 简化 properties
    props = schema.get("properties")
    if isinstance(props, dict):
        simplified_props = {}
        for name, prop_schema in props.items():
            if not isinstance(prop_schema, dict):
                simplified_props[name] = prop_schema
                continue

            simplified_prop = {}

            # 保留 type
            if "type" in prop_schema:
                simplified_prop["type"] = prop_schema["type"]

            # 递归简化嵌套 properties（object 类型）
            if "properties" in prop_schema:
                nested_schema = {"type": "object", "properties": prop_schema["properties"]}
                if "required" in prop_schema:
                    nested_schema["required"] = prop_schema["required"]
                if "additionalProperties" in prop_schema:
                    nested_schema["additionalProperties"] = prop_schema["additionalProperties"]
                nested = _simplify_json_schema(nested_schema)
                if "properties" in nested:
                    simplified_prop["properties"] = nested["properties"]
                if "required" in nested:
                    simplified_prop["required"] = nested["required"]
                if "additionalProperties" in nested:
                    simplified_prop["additionalProperties"] = nested["additionalProperties"]

            # 保留 items（数组类型）
            if "items" in prop_schema:
                simplified_prop["items"] = _simplify_json_schema(prop_schema["items"])

            # 保留 enum
            if "enum" in prop_schema:
                simplified_prop["enum"] = prop_schema["enum"]

            simplified_props[name] = simplified_prop

        result["properties"] = simplified_props

    return result
