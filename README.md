<p align="center">
  <img src="assets/icon.svg" width="80" height="96" alt="Kiro Proxy">
</p>

<h1 align="center">Kiro API Proxy</h1>

<p align="center">
  Kiro IDE API 反向代理服务器，支持多账号轮询、Token 自动刷新、配额管理
</p>

<p align="center">
  <a href="#功能特性">功能</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#cli-配置">CLI 配置</a> •
  <a href="#api-端点">API</a> •
  <a href="#许可证">许可证</a>
</p>

---

> **⚠️ 测试说明**
> 
> 本项目支持 **Claude Code**、**Codex CLI**、**Gemini CLI** 三种客户端，工具调用功能已全面支持。

## 功能特性

### 核心功能
- **多协议支持** - OpenAI / Anthropic / Gemini 三种协议兼容
- **完整工具调用** - 三种协议的工具调用功能全面支持
- **多账号轮询** - 支持添加多个 Kiro 账号，自动负载均衡
- **会话粘性** - 同一会话 60 秒内使用同一账号，保持上下文
- **Web UI** - 简洁的管理界面，支持监控、日志、设置

### v1.6.1 新功能
- **请求限速** - 通过限制请求频率降低账号封禁风险
  - 每账号最小请求间隔
  - 每账号每分钟最大请求数
  - 全局每分钟最大请求数
  - WebUI 设置页面可配置
- **账号封禁检测** - 自动检测 TEMPORARILY_SUSPENDED 错误
  - 友好的错误日志输出
  - 自动禁用被封禁账号
  - 自动切换到其他可用账号
- **统一错误处理** - 三种协议使用统一的错误分类和处理

### v1.6.0 功能
- **历史消息管理** - 4 种策略处理对话长度限制，可自由组合
  - 自动截断：发送前按数量/字符数截断
  - 智能摘要：用 AI 生成早期对话摘要，保留关键信息
  - 错误重试：遇到长度错误时自动截断重试（默认启用）
  - 预估检测：预估 token 数量，超限预先截断
- **Gemini 工具调用** - 完整支持 functionDeclarations/functionCall/functionResponse
- **设置页面** - WebUI 新增设置标签页，可配置历史消息管理策略

### v1.5.0 功能
- **用量查询** - 查询账号配额使用情况，显示已用/余额/使用率
- **多登录方式** - 支持 Google / GitHub / AWS Builder ID 三种登录方式
- **流量监控** - 完整的 LLM 请求监控，支持搜索、过滤、导出
- **浏览器选择** - 自动检测已安装浏览器，支持无痕模式
- **文档中心** - 内置帮助文档，左侧目录 + 右侧 Markdown 渲染

### v1.4.0 功能
- **Token 预刷新** - 后台每 5 分钟检查，提前 15 分钟自动刷新
- **健康检查** - 每 10 分钟检测账号可用性，自动标记状态
- **请求统计增强** - 按账号/模型统计，24 小时趋势
- **请求重试机制** - 网络错误/5xx 自动重试，指数退避

## 工具调用支持

| 功能 | Anthropic (Claude Code) | OpenAI (Codex CLI) | Gemini |
|------|------------------------|-------------------|--------|
| 工具定义 | ✅ `tools` | ✅ `tools.function` | ✅ `functionDeclarations` |
| 工具调用响应 | ✅ `tool_use` | ✅ `tool_calls` | ✅ `functionCall` |
| 工具结果 | ✅ `tool_result` | ✅ `tool` 角色消息 | ✅ `functionResponse` |
| 强制工具调用 | ✅ `tool_choice` | ✅ `tool_choice` | ✅ `toolConfig.mode` |
| 工具数量限制 | ✅ 50 个 | ✅ 50 个 | ✅ 50 个 |
| 历史消息修复 | ✅ | ✅ | ✅ |

## 已知限制

### 对话长度限制

Kiro API 有输入长度限制。当对话历史过长时，会返回错误：

```
Input is too long. (CONTENT_LENGTH_EXCEEDS_THRESHOLD)
```

#### 自动处理（v1.6.0+）

代理内置了历史消息管理功能，可在「设置」页面配置：

- **错误重试**（默认）：遇到长度错误时自动截断并重试
- **智能摘要**：用 AI 生成早期对话摘要，保留关键信息
- **自动截断**：每次请求前按数量/字符数截断
- **预估检测**：预估 token 数量，超限预先截断

#### 手动处理

1. 在 Claude Code 中输入 `/clear` 清空对话历史
2. 告诉 AI 你之前在做什么，它会读取代码文件恢复上下文

## 快速开始

### 方式一：下载预编译版本

从 [Releases](../../releases) 下载对应平台的安装包，解压后直接运行。

### 方式二：从源码运行

```bash
# 克隆项目
git clone https://github.com/yourname/kiro-proxy.git
cd kiro-proxy

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 运行
python run.py

# 或指定端口
python run.py 8081
```

启动后访问 http://localhost:8080

### 登录获取 Token

**方式一：在线登录（推荐）**
1. 打开 Web UI，点击「在线登录」
2. 选择登录方式：Google / GitHub / AWS Builder ID
3. 在浏览器中完成授权
4. 账号自动添加

**方式二：扫描 Token**
1. 打开 Kiro IDE，使用 Google/GitHub 账号登录
2. 登录成功后 token 自动保存到 `~/.aws/sso/cache/`
3. 在 Web UI 点击「扫描 Token」添加账号

## CLI 配置

### 模型对照表

| Kiro 模型 | 能力 | Claude Code | Codex |
|-----------|------|-------------|-------|
| `claude-sonnet-4` | ⭐⭐⭐ 推荐 | `claude-sonnet-4` | `gpt-4o` |
| `claude-sonnet-4.5` | ⭐⭐⭐⭐ 更强 | `claude-sonnet-4.5` | `gpt-4o` |
| `claude-haiku-4.5` | ⚡ 快速 | `claude-haiku-4.5` | `gpt-4o-mini` |
| `claude-opus-4.5` | ⭐⭐⭐⭐⭐ 最强 | `claude-opus-4.5` | `o1` |

### Claude Code 配置

```
名称: Kiro Proxy
API Key: any
Base URL: http://localhost:8080
模型: claude-sonnet-4
```

### Codex 配置

```
名称: Kiro Proxy
API Key: any
Endpoint: http://localhost:8080/v1
模型: gpt-4o
```

## API 端点

| 协议 | 端点 | 用途 |
|------|------|------|
| OpenAI | `POST /v1/chat/completions` | Codex CLI |
| OpenAI | `GET /v1/models` | 模型列表 |
| Anthropic | `POST /v1/messages` | Claude Code |
| Anthropic | `POST /v1/messages/count_tokens` | Token 计数 |
| Gemini | `POST /v1/models/{model}:generateContent` | Gemini CLI |

### 管理 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/accounts` | GET | 获取所有账号状态 |
| `/api/accounts/{id}` | GET | 获取账号详情 |
| `/api/accounts/{id}/usage` | GET | 获取账号用量信息 |
| `/api/accounts/{id}/refresh` | POST | 刷新账号 Token |
| `/api/accounts/{id}/restore` | POST | 恢复账号（从冷却状态） |
| `/api/accounts/refresh-all` | POST | 刷新所有即将过期的 Token |
| `/api/flows` | GET | 获取流量记录 |
| `/api/flows/stats` | GET | 获取流量统计 |
| `/api/flows/{id}` | GET | 获取流量详情 |
| `/api/quota` | GET | 获取配额状态 |
| `/api/stats` | GET | 获取统计信息 |
| `/api/health-check` | POST | 手动触发健康检查 |
| `/api/browsers` | GET | 获取可用浏览器列表 |
| `/api/docs` | GET | 获取文档列表 |
| `/api/docs/{id}` | GET | 获取文档内容 |

## 项目结构

```
kiro_proxy/
├── main.py                    # FastAPI 应用入口
├── config.py                  # 全局配置
├── converters.py              # 协议转换
│
├── core/                      # 核心模块
│   ├── account.py            # 账号管理
│   ├── state.py              # 全局状态
│   ├── persistence.py        # 配置持久化
│   ├── scheduler.py          # 后台任务调度
│   ├── stats.py              # 请求统计
│   ├── retry.py              # 重试机制
│   ├── browser.py            # 浏览器检测
│   ├── flow_monitor.py       # 流量监控
│   └── usage.py              # 用量查询
│
├── credential/                # 凭证管理
│   ├── types.py              # KiroCredentials
│   ├── fingerprint.py        # Machine ID 生成
│   ├── quota.py              # 配额管理器
│   └── refresher.py          # Token 刷新
│
├── auth/                      # 认证模块
│   └── device_flow.py        # Device Code Flow / Social Auth
│
├── handlers/                  # API 处理器
│   ├── anthropic.py          # /v1/messages
│   ├── openai.py             # /v1/chat/completions
│   ├── gemini.py             # /v1/models/{model}:generateContent
│   └── admin.py              # 管理 API
│
├── docs/                      # 内置文档
│   ├── 01-quickstart.md      # 快速开始
│   ├── 02-features.md        # 功能特性
│   ├── 03-faq.md             # 常见问题
│   └── 04-api.md             # API 参考
│
└── web/
    └── html.py               # Web UI (组件化单文件)
```

## 构建

```bash
# 安装构建依赖
pip install pyinstaller

# 构建
python build.py
```

输出文件在 `dist/` 目录。

## 免责声明

本项目仅供学习研究，禁止商用。使用本项目产生的任何后果由使用者自行承担，与作者无关。

本项目与 Kiro / AWS / Anthropic 官方无关。
