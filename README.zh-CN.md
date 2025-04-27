# Poe Bot Host

[English](README.md) | 简体中文

一个用于 Poe 平台的多机器人框架，简化了机器人的创建、部署和管理。

```
poe_bots/
├── 🌐 app.py          # 主 FastAPI 应用程序
├── 🤖 bots/           # 机器人实现
│   ├── echo_bot.py    # 简单回声机器人
│   ├── weather_bot.py # 天气信息机器人
│   └── ...            # 其他特色机器人
├── 📘 examples/       # 示例代码和指南
├── 🧪 tests/          # 全面的测试套件
└── 🛠️ utils/          # 核心工具
    ├── api_keys.py    # API 密钥管理
    ├── base_bot.py    # 基础机器人架构
    └── bot_factory.py # 机器人注册系统
```

## 🚀 快速入门

```bash
# 克隆和安装
git clone https://github.com/YushengAuggie/poe-bot-host.git
cd poe-bot-host
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 本地运行
./run_local.sh

# 测试机器人
python test_bot.py --bot EchoBot --message "你好世界！"
```

需要更多详细信息？查看我们的[5分钟快速入门指南](QUICKSTART.md)。

## 📚 文档

- [**QUICKSTART.md**](QUICKSTART.md): 5分钟内入门
- [**DEPLOYMENT.md**](DEPLOYMENT.md): 分步部署指南
- [**Examples**](examples/): 示例机器人和实现指南

## 🏗️ 项目架构

Poe Bot Host 由以下主要组件组织：

### 核心框架

| 组件 | 目的 |
|-----------|---------|
| **app.py** | 用于机器人托管的主 FastAPI 应用程序 |
| **utils/** | 机器人管理的核心工具 |
| **run_local.py/.sh** | 本地开发服务器 |

### 机器人实现

| 组件 | 目的 |
|-----------|---------|
| **BaseBot** | 具有通用功能的抽象基类 |
| **BotFactory** | 自动发现并注册所有机器人 |
| **bots/** | 可直接使用的机器人实现 |

### 包含的机器人类型

- **基础机器人**: Echo, Reverse, Uppercase
- **高级机器人**: BotCaller, Weather, WebSearch  
- **功能性机器人**: Calculator, FunctionCalling, FileAnalyzer

## 🔌 简化的 API 密钥管理

本框架提供了一种简化的 API 密钥管理系统，用于与外部服务集成：

```python
from utils.api_keys import get_api_key

# 从环境变量或 Modal 密钥获取 API 密钥
openai_key = get_api_key("OPENAI_API_KEY")
google_key = get_api_key("GOOGLE_API_KEY")
custom_key = get_api_key("CUSTOM_SERVICE_API_KEY")
```

### 工作原理
1. 首先检查环境变量（用于本地开发）
2. 然后检查 Modal 密钥（用于云部署）
3. 如果找不到密钥，则引发清晰的错误消息

### 机器人集成示例
```python
# 在您的机器人实现中
def get_client():
    try:
        return OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
    except Exception as e:
        logger.warning(f"初始化客户端失败：{str(e)}")
        return None
```

查看[API 密钥管理指南](examples/api_key_management.md)获取完整文档。

## 🛠️ 开发工作流程

1. **创建机器人**: 复制并修改 `bots/template_bot.py`
2. **本地测试**: `./run_local.sh --debug`
3. **验证**: `python test_bot.py --bot YourBot`
4. **部署**: `modal deploy app.py`
5. **在 Poe 上配置**: 连接到您的 Modal 端点

## 🌟 创建您的第一个机器人

```python
from fastapi_poe.types import PartialResponse, QueryRequest
from utils.base_bot import BaseBot

class MyAwesomeBot(BaseBot):
    bot_name = "MyAwesomeBot"
    bot_description = "做一些很棒的事情"

    async def get_response(self, query: QueryRequest):
        user_message = self._extract_message(query)
        response = f"您说：{user_message}"
        yield PartialResponse(text=response)
```

有关更多详细信息，请参阅下面的[创建新机器人](#创建新机器人)部分。

## 📋 完整文档

以下部分包含框架的完整文档。

---

## 本地运行平台

您可以在本地运行平台进行开发和测试。有两种等效的方式启动服务器：

### 使用 Shell 脚本

Shell 脚本在运行前会自动激活您的虚拟环境：

```bash
# 基本运行
./run_local.sh

# 开发模式，自动重载
./run_local.sh --reload

# 调试模式，详细日志
./run_local.sh --debug

# 自定义端口
./run_local.sh --port 9000

# 所有选项组合
./run_local.sh --debug --reload --port 9000 --host 127.0.0.1
```

### 直接使用 Python

如果您更喜欢直接使用 Python（确保虚拟环境已激活）：

```bash
python run_local.py --debug --reload
```

### 可用选项

两种方法支持相同的选项：

| 选项 | 描述 |
|--------|-------------|
| `--host` | 绑定的主机（默认：0.0.0.0） |
| `--port` | 绑定的端口（默认：8000） |
| `--reload` | 开发时启用自动重载 |
| `--debug` | 启用调试模式，详细日志 |
| `--log-level` | 设置日志级别（DEBUG、INFO、WARNING、ERROR、CRITICAL） |
| `--no-banner` | 不显示横幅 |
| `--help` | 显示帮助信息并退出 |

默认情况下，服务器将在 http://localhost:8000 启动，所有发现的机器人均可用。

## 测试机器人

平台包含一个全面的测试工具：

```bash
# 测试第一个可用机器人
python test_bot.py

# 测试特定机器人
python test_bot.py --bot EchoBot

# 使用自定义消息测试
python test_bot.py --bot ReverseBot --message "反转这段文本"

# 检查 API 健康状况
python test_bot.py --health

# 显示 API 端点
python test_bot.py --schema
```

### 使用 curl 手动测试

要使用 curl 手动测试您的机器人，首先确保您的服务器正在运行：

```bash
# 在一个终端中启动服务器
source venv/bin/activate  # 确保您的虚拟环境已激活
./run_local.sh  # 或 python run_local.py
```

然后在另一个终端中，您可以发送请求：

```bash
# 获取可用机器人列表
curl http://localhost:8000/bots

# 检查 API 健康状况
curl http://localhost:8000/health

# 测试特定机器人（将 echobot 替换为您的机器人小写名称）
curl -X POST "http://localhost:8000/echobot" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummytoken" \
  -d '{
    "version": "1.0",
    "type": "query",
    "query": [
        {"role": "user", "content": "你好世界！"}
    ],
    "user_id": "test_user",
    "conversation_id": "test_convo_123",
    "message_id": "test_msg_123",
    "protocol": "poe"
  }'
```

## 创建新机器人

### 使用模板

创建新机器人最简单的方法是复制并修改模板：

1. 将 `bots/template_bot.py` 复制到 `bots/your_bot_name.py`
2. 修改类名、机器人名称和描述
3. 在 `get_response` 方法中实现您的逻辑

示例：
```python
import json
from typing import AsyncGenerator, Union
from fastapi_poe.types import PartialResponse, QueryRequest, MetaResponse
from utils.base_bot import BaseBot

class WeatherBot(BaseBot):
    """提供天气信息的机器人。"""

    bot_name = "WeatherBot"
    bot_description = "为指定位置提供天气信息"
    version = "1.0.0"

    async def get_response(self, query: QueryRequest) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        # 提取用户消息
        user_message = self._extract_message(query)

        # 处理机器人信息请求
        if user_message.lower().strip() == "bot info":
            metadata = self._get_bot_metadata()
            yield PartialResponse(text=json.dumps(metadata, indent=2))
            return

        # 解析消息获取位置
        location = user_message.strip()

        # 在真实机器人中，您会调用天气 API
        weather_info = f"{location}的天气晴朗，最高温度为24°C。"

        # 返回响应
        yield PartialResponse(text=weather_info)
```

### 机器人配置选项

您的机器人可以包含各种配置选项：

```python
class ConfigurableBot(BaseBot):
    bot_name = "ConfigurableBot"
    bot_description = "具有自定义配置的机器人"
    version = "1.0.0"

    # 自定义设置
    max_message_length = 5000  # 覆盖默认值（2000）
    stream_response = False    # 禁用流式响应

    # 您也可以添加自己的设置
    api_key = "default-key"   # 自定义设置

    def __init__(self, **kwargs):
        # 使用环境变量或 kwargs 初始化
        settings = {
            "api_key": os.environ.get("MY_BOT_API_KEY", self.api_key)
        }
        super().__init__(settings=settings, **kwargs)
```

### 错误处理

框架提供了内置的错误处理，有两种错误类型：

- `BotError`：可重试的常规错误
- `BotErrorNoRetry`：不应重试的错误

用法示例：

```python
from utils.base_bot import BaseBot, BotError, BotErrorNoRetry

class ErrorHandlingBot(BaseBot):
    # ...

    async def get_response(self, query: QueryRequest):
        try:
            # 提取用户消息
            message = self._extract_message(query)

            # 可能失败的代码
            if not self._is_valid_input(message):
                # 用户错误 - 不要重试
                raise BotErrorNoRetry("输入格式无效。请尝试其他内容。")

            result = await self._fetch_external_data(message)
            if not result:
                # 服务错误 - 可重试
                raise BotError("服务不可用。请稍后再试。")

            yield PartialResponse(text=result)

        except Exception as e:
            # 处理意外错误
            self.logger.error(f"意外错误：{str(e)}", exc_info=True)
            raise  # 框架将处理
```

## 部署机器人

### 分步部署到 Modal

[Modal](https://modal.com/) 是一个运行 Python 代码的云平台。以下是将 Poe 机器人部署到 Modal 的详细指南：

#### 1. 设置 Modal

1. 在 [modal.com](https://modal.com/signup) 注册 Modal 账户

2. 在环境中安装 Modal 客户端：
   ```bash
   pip install modal-client
   ```

3. 通过运行以下命令进行 Modal 认证：
   ```bash
   modal token new
   ```
   - 这将打开浏览器窗口
   - 登录您的 Modal 账户
   - CLI 将自动保存您的认证令牌

#### 2. 部署您的机器人

要部署包含所有机器人的整个框架：

```bash
modal deploy app.py
```

您将看到类似这样的输出：

```
Modal app poe-bots deployed!
Endpoints:
→ web_endpoint: https://yourusername--poe-bots-fastapi-app.modal.run
```

记下这个 URL - 您需要它来在 Poe 上配置机器人。

如果您想要部署独立机器人（使用 example_standalone_bot.py），运行：

```bash
modal deploy examples/standalone_echobot.py
```

这将输出该独立机器人的特定 URL。

#### 3. 测试您的部署

在 Poe 上配置之前，测试您的部署是否正常工作：

```bash
# 测试健康端点
curl https://yourusername--poe-bots-fastapi-app.modal.run/health

# 列出可用机器人
curl https://yourusername--poe-bots-fastapi-app.modal.run/bots
```

您应该看到一个成功的响应，包含健康信息和可用机器人列表。

### 在 Poe 上设置机器人

部署 API 后，您可以在 Poe 上创建连接到您 API 的机器人：

#### 1. 在 Poe 上创建机器人

1. 前往 [Poe Creator Portal](https://creator.poe.com/)
2. 点击"Create Bot"（或"Create a bot"按钮）
3. 填写基本详情：
   - **Name**：您的机器人的唯一名称（例如"EchoBot"）
   - **Description**：清晰描述您的机器人功能
   - **Instructions/Prompt**：机器人的可选指令

#### 2. 配置服务器设置

1. 在机器人创建表单中，向下滚动到"Server Bot Settings"，选择"Server bot"作为机器人类型
2. 配置 API：
   - **Server Endpoint**：您的 Modal 部署 URL + 特定机器人路径
     - 格式：`https://yourusername--poe-bots-fastapi-app.modal.run/botname`
     - EchoBot 示例：`https://yourusername--poe-bots-fastapi-app.modal.run/echobot`
     - 注意路径是您机器人名称的小写版本
   - **API Protocol**：选择"Poe Protocol"
   - **API Key Protection**：选择"No protection"（或如果您设置了 API 密钥，请配置它）

#### 3. 添加机器人元数据（可选但推荐）

为了获得更完善的机器人体验，添加：

1. **Sample Messages**：添加 3-5 个示例消息，帮助用户知道要问什么
   - EchoBot 示例："你好世界"，"回显这条消息"，"跟我重复：测试"
   - WeatherBot 示例："纽约天气"，"伦敦的天气预报是什么？"，"东京天气"

2. **Knowledge Files**：如果您的机器人需要参考材料，可以在这里上传
   - 本框架中的大多数机器人不需要这个，因为它们直接处理输入

3. **API Citation Preference**：选择您的机器人应如何引用来源
   - 对于本框架中的大多数机器人，选择"Don't cite sources"比较合适

4. **Message feedback**：选择是否允许用户反馈（推荐用于收集改进想法）

#### 4. 保存并测试

1. 点击"Create Bot"保存您的配置
2. 创建后，您将被重定向到与您的机器人的聊天
3. 通过发送消息测试您的机器人
4. 消息将被发送到您的 Modal 托管 API，由您的机器人处理，响应将显示在聊天中

如果您的机器人没有响应或遇到错误，请参考下面的故障排除部分。

## 维护和开发

本框架专为便于维护和扩展而设计。以下是如何保持其顺畅运行并添加新功能。

### 版本管理

当前版本为 1.0.0。进行重大更改时：

1. 更新以下位置的版本号：
   - `pyproject.toml`
   - `app.py`（`__version__` 变量）
   - 任何文档引用

2. 遵循语义化版本规则：
   - MAJOR：破坏性更改
   - MINOR：新功能，向后兼容
   - PATCH：错误修复，向后兼容

### 添加新功能

向框架添加新功能：

1. 对于机器人特定功能：
   - 在 `utils/base_bot.py` 中添加方法
   - 使用文档字符串清晰记录它们
   - 使用类型提示以获得更好的 IDE 支持

2. 对于平台功能：
   - 在 `app.py` 中添加端点
   - 更新 `utils/bot_factory.py` 中的 `BotFactory`
   - 更新测试以覆盖新功能

3. 对于配置选项：
   - 添加到 `utils/config.py`
   - 使用新选项更新 `.env.example`

### 日志和监控

框架使用 Python 内置的日志功能。配置日志级别：

- 在代码中：`logger.setLevel(logging.DEBUG)`
- 通过环境：`DEBUG=true ./run_local.sh`
- 通过 CLI：`./run_local.sh --debug`

对于生产监控：
- 使用 Modal 内置的仪表板
- 考虑添加结构化日志以便于分析
- 为关键错误设置警报

### 测试更改

始终测试您的更改：

1. 运行本地服务器：`./run_local.sh --debug`
2. 测试所有机器人：`python test_bot.py --all`
3. 测试特定更改：`python test_bot.py --bot YourBot --message "测试消息"`
4. 运行自动化测试：`make test`
5. 检查代码规范：`make lint`
6. 验证格式：`make format`

### 最佳实践

1. **机器人设计**：
   - 保持机器人专注于单一任务
   - 使用清晰、描述性的名称和文档
   - 通过特定错误消息优雅地处理错误
   - 在响应中考虑用户体验

2. **代码组织**：
   - 遵循 Python 的 PEP 8 风格指南
   - 使用类型提示以获得更好的 IDE 支持
   - 记录所有公共方法和类
   - 保持相关功能在一起

3. **性能**：
   - 对 I/O 绑定操作使用异步函数
   - 尽可能保持机器人无状态
   - 避免不必要的依赖
   - 对重复操作考虑缓存

4. **安全**：
   - 绝不在代码中存储凭证
   - 对敏感信息使用环境变量
   - 验证所有用户输入
   - 保持依赖更新

## 故障排除

### 常见问题

1. **找不到机器人**：
   - 确保机器人类继承自 `BaseBot`
   - 检查文件是否在 `bots` 目录中
   - 验证机器人名称是唯一的

2. **部署错误**：
   - 检查 Modal 凭证：`modal token new`
   - 验证 `requirements.txt` 中的要求

3. **运行时错误**：
   - 使用调试模式运行：`./run_local.sh --debug`
   - 检查日志以获取特定错误消息

## 持续集成与质量保证

本项目使用 GitHub Actions 在每次推送和拉取请求时运行自动化测试和代码质量检查。

### CI/CD 流水线

CI/CD 流水线运行：
- 在多个 Python 版本上使用 pytest 进行单元测试
- 使用 ruff 进行代码规范检查
- 使用 pyright 进行类型检查

您可以在仓库的 GitHub Actions 标签页中查看 CI 流水线的状态。

### Pre-Commit 钩子

Pre-commit 钩子用于确保代码质量，在每次提交和推送前自动检查您的代码。

#### 设置说明

```bash
# 安装 pre-commit 工具
pip install pre-commit

# 安装 pre-commit 钩子
pre-commit install --install-hooks

# 同时安装 pre-push 钩子（用于测试）
pre-commit install --hook-type pre-push
```

#### 钩子检查内容

每次提交时：
- **代码规范检查**（ruff）：检查代码风格和格式
- **类型检查**（pyright）：验证正确的类型使用
- **安全检查**：检测私钥、调试语句等
- **文件格式化**：修复尾随空格、行尾等

每次推送时：
- **测试**（pytest）：运行整个测试套件

#### 好处

- 防止提交有错误或质量差的代码
- 提供问题的即时反馈
- 确保团队间的代码质量一致
- 减少关于风格/格式的代码审查评论
- 一些问题会被自动修复

## 资源

- [Poe 文档](https://creator.poe.com/docs)
- [fastapi-poe 文档](https://github.com/poe-platform/fastapi-poe)
- [Modal 文档](https://modal.com/docs)
- [FastAPI 文档](https://fastapi.tiangolo.com/)

## 许可证

本项目采用 MIT 许可证授权 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 作者

为 Poe 社区创建，充满❤️。

---

*本框架与 Poe、Modal 或 Anthropic 没有官方关联。*
