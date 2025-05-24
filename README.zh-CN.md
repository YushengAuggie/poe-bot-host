# 🤖 Poe Bot Host

[English](README.md) | 简体中文

一个**现代化、生产就绪**的 Poe 平台多机器人托管框架。使用清洁架构和企业级可靠性来构建、部署和管理高级 AI 机器人。

```
poe_bots/
├── 🌐 app.py                    # 带自动发现的 FastAPI 应用程序
├── 🤖 bots/                     # 即用型机器人实现
│   ├── calculator_bot.py        # ➕ 数学计算与单位转换
│   ├── echo_bot.py              # 🔄 带错误处理的简单回声
│   ├── weather_bot.py           # 🌤️ 天气信息与 API 集成
│   ├── chatgpt.py               # 💬 OpenAI GPT 集成
│   ├── gemini*.py               # 🧠 Google Gemini AI（多模态支持）
│   └── web_search_bot.py        # 🔍 网络搜索功能
├── 🛠️ utils/                    # 现代化、模块化架构
│   ├── mixins.py                # 🎯 可重用的错误处理和响应模式
│   ├── auth.py                  # 🔐 智能 API 密钥解析与回退
│   ├── calculators.py           # 🔢 安全数学运算与转换
│   ├── media/                   # 📁 媒体处理框架
│   │   ├── processors.py        # 🖼️ 图像/视频/音频附件处理
│   │   ├── validators.py        # ✅ 媒体类型与大小验证
│   │   └── converters.py        # 🔄 格式转换与优化
│   ├── base_bot.py              # 🏗️ 带自动发现的增强基类
│   └── bot_factory.py           # 🏭 智能机器人注册与管理
├── 📘 examples/                 # 全面的指南与示例
├── 🧪 tests/                    # 全面的自动化测试套件
└── 🚀 简易部署                  # 一键部署到 Modal/云端
```

## ✨ 为什么选择此框架？

🎯 **开发者优先设计**: 清洁架构、优秀文档、全面测试
🔧 **企业就绪**: 强大错误处理、安全 API 密钥管理、生产级日志
🚀 **分钟内部署**: 一键部署到 Modal，支持自动缩放
🧩 **模块化与可扩展**: 可重用组件、领域专用工具、清洁抽象
⚡ **高性能**: 优化媒体处理、智能缓存、最小开销
🛡️ **安全优先**: 输入验证、安全表达式求值、API 密钥加密

## 🚀 快速开始（2分钟）

```bash
# 1. 克隆和设置
git clone https://github.com/YushengAuggie/poe-bot-host.git
cd poe-bot-host
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. 本地运行（自动启动所有机器人）
./run_local.sh

# 3. 测试您的机器人
python scripts/test_bot_cli.py --bot CalculatorBot --message "2 + 2"
python scripts/test_bot_cli.py --bot EchoBot --message "你好世界！"

# 4. 部署到生产环境（一条命令！）
modal deploy app.py
```

**就是这样！** 您的机器人现在运行在 `http://localhost:8000` 🎉

## 🏗️ 现代化架构

### 🎯 **清晰的关注点分离**

我们重构的架构遵循企业模式，最大化可维护性：

| 组件 | 目的 | 好处 |
|-----------|---------|----------|
| **🤖 机器人层** | 业务逻辑与用户交互 | 易于添加新机器人，职责清晰 |
| **🎭 混入层** | 可重用模式（错误、响应） | DRY 原则，跨机器人一致的用户体验 |
| **🛠️ 工具层** | 领域专用工具 | 可测试、可重用、专注的模块 |
| **🌐 应用层** | HTTP 路由与机器人发现 | 自动发现、健康检查、监控 |

### 🧩 **模块化工具** (新功能!)

**媒体处理框架**
```python
from utils.media import MediaProcessor, MediaValidator

processor = MediaProcessor()
attachments = processor.extract_attachments(query)  # 自动处理图像/视频/音频
media_parts = processor.prepare_media_parts(attachments)  # 为 AI 模型准备
```

**智能身份验证**
```python
from utils.auth import APIKeyResolver

resolver = APIKeyResolver("MyBot")
api_key = resolver.resolve()  # 自动尝试多种命名模式
```

**安全计算**
```python
from utils.calculators import SafeExpressionEvaluator, UnitConverter

evaluator = SafeExpressionEvaluator()
result = evaluator.evaluate("sin(30) + log(100)")  # 防止代码注入

converter = UnitConverter()
celsius = converter.convert(32, "f_to_c")  # (0.0, "°C")
```

### 🎯 **增强错误处理**
```python
from utils.mixins import ErrorHandlerMixin, ResponseMixin

class MyBot(BaseBot, ErrorHandlerMixin, ResponseMixin):
    async def get_response(self, query):
        async def _process():
            # 您的机器人逻辑
            yield PartialResponse(text="你好！")

        # 自动错误处理，支持重试和用户友好消息
        async for response in self.handle_common_errors(query, _process):
            yield response
```

## 🤖 创建您的第一个机器人（30秒）

```python
# bots/my_awesome_bot.py
from utils.base_bot import BaseBot
from utils.mixins import ErrorHandlerMixin, ResponseMixin
from fastapi_poe.types import PartialResponse, QueryRequest

class MyAwesomeBot(BaseBot, ErrorHandlerMixin, ResponseMixin):
    bot_name = "MyAwesomeBot"
    bot_description = "做一些令人惊叹的事情！"

    async def get_response(self, query: QueryRequest):
        async def _process():
            user_message = self._extract_message(query)

            if user_message.lower() == "help":
                help_text = "我可以帮助您做令人惊叹的事情！\n尝试：'你好'、'笑话'、'建议'"
                yield PartialResponse(text=self._format_help_response(help_text))
                return

            if "你好" in user_message.lower():
                yield PartialResponse(text="👋 你好！我是您的超赞机器人！")
            elif "笑话" in user_message.lower():
                yield PartialResponse(text="🤖 为什么机器人要过马路？为了到达另一个字节！")
            else:
                yield PartialResponse(text=f"✨ 您说：'{user_message}' - 太棒了！")

        async for response in self.handle_common_errors(query, _process):
            yield response
```

**您免费获得的功能：**
- ✅ 自动错误处理与用户友好消息
- ✅ 带表情符号支持的一致响应格式
- ✅ API 密钥管理（如需要）
- ✅ 帮助命令格式化
- ✅ 日志记录和调试
- ✅ 框架自动发现

## 🔑 简化的 API 密钥管理

### **两种密钥类型**
1. **🔌 服务 API 密钥**（OpenAI、Google 等）- 用于外部服务
2. **🤖 机器人访问密钥** - 每个机器人的唯一 Poe 密钥（从 Poe 仪表板获取）

### **设置（一次性）**
在项目根目录创建 `.env` 文件：
```bash
# 服务 API 密钥（可选 - 仅当您的机器人需要时）
OPENAI_API_KEY=sk-...您的openai密钥...
GOOGLE_API_KEY=AIza...您的google密钥...

# 来自 Poe 的机器人访问密钥（部署时必需）
ECHO_BOT_ACCESS_KEY=psk_...来自poe仪表板...
CALCULATOR_BOT_ACCESS_KEY=psk_...来自poe仪表板...
MYAWESOME_BOT_ACCESS_KEY=psk_...来自poe仪表板...
```

### **灵活命名**
框架使用多种模式自动查找您的密钥：
```bash
# 对于 "WeatherBot"，以下都有效：
WEATHER_BOT_ACCESS_KEY=psk_...     # ✅ 推荐
WEATHERBOT_ACCESS_KEY=psk_...      # ✅ 也可以
WeatherBot_ACCESS_KEY=psk_...      # ✅ 也可以
```

### **获取机器人访问密钥**
1. 访问 https://poe.com/edit_bot?bot=YOUR_BOT_NAME
2. 从页面复制访问密钥
3. 添加到您的 `.env` 文件

**就是这样！** 框架自动处理其他一切。

## 🚀 部署（生产就绪）

### **部署到 Modal（推荐）**
```bash
# 一键部署
modal deploy app.py

# 输出示例：
# ✅ Modal 应用已部署！
# 🌐 https://yourname--poe-bots-fastapi-app.modal.run
```

### **连接到 Poe**
1. **创建机器人** 在 [Poe Creator Portal](https://creator.poe.com/)
2. **配置服务器设置**：
   - **端点**: `https://yourname--poe-bots-fastapi-app.modal.run/mybotname`
   - **协议**: Poe Protocol
   - **保护**: 无保护（或配置 API 密钥）
3. **测试**: 发送消息给您的机器人！

### **关键功能**
- ✅ **易于部署**: 一键部署到 Modal
- ✅ **健康监控**: 内置健康检查端点
- ✅ **错误处理**: 全面的错误日志记录
- ✅ **灵活托管**: 本地或云端部署

## 🛠️ 开发工作流

### **本地开发**
```bash
# 启动带自动重载的开发服务器
./run_local.sh --debug --reload

# 测试特定机器人
python scripts/test_bot_cli.py --bot YourBot --message "测试消息"

# 检查所有机器人是否正常工作
python scripts/test_bot_cli.py --health
```

### **测试与质量**
```bash
# 运行所有 257 个自动化测试
pytest tests/ -v

# 检查代码格式
black . && ruff check .

# 类型检查
pyright .
```

### **持续集成**
- ✅ **预提交钩子**: 自动代码格式化和代码检查
- ✅ **GitHub Actions**: 每次推送自动测试
- ✅ **质量门禁**: 类型检查、安全扫描、测试覆盖率

## 📚 高级功能

### **🖼️ 多模态支持（图像/视频/音频）**
```python
# 内置媒体附件支持
attachments = self.media_processor.extract_attachments(query)
if attachments:
    for attachment in attachments:
        # 处理图像、视频或音频文件
        processed = self.media_processor.process_attachment(attachment)
```

### **🔒 安全表达式求值**
```python
# 安全的数学计算（无代码注入）
from utils.calculators import SafeExpressionEvaluator

evaluator = SafeExpressionEvaluator()
result = evaluator.evaluate("2 + 2 * sin(30)")  # ✅ 安全
result = evaluator.evaluate("import os; os.system('rm -rf /')")  # ❌ 被阻止
```

### **🎯 响应格式化**
```python
# 一致、美观的响应
self._format_help_response("您的帮助文本")  # 🤖 机器人名称 帮助
self._format_error_response("出了点问题")  # ❌ 错误：出了点问题
self._format_success_response("任务完成")  # ✅ 任务完成
```

### **📊 内置分析**
- 通过 `/health` 端点提供机器人使用指标
- 性能监控
- 错误率跟踪
- 响应时间分析

## 🧪 示例机器人

框架包含多个即用型机器人：

- **基础机器人**: 回声、计算器、天气
- **AI 驱动机器人**: ChatGPT、Gemini（支持多模态）
- **工具机器人**: 网络搜索、文件分析、函数调用
- **模板机器人**: 用作新机器人的起点

浏览 `bots/` 目录查看所有可用实现。

## 🆘 故障排除

### **常见问题**

**🔍 找不到机器人**
```bash
# 检查机器人是否被发现
python scripts/test_bot_cli.py --list-bots

# 验证机器人类继承自 BaseBot
class MyBot(BaseBot):  # ✅ 正确
```

**🔐 身份验证错误**
```bash
# 检查 API 密钥是否已加载
python -c "import os; print([k for k in os.environ if 'API_KEY' in k])"

# 验证 .env 文件存在并已加载
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('YOUR_KEY_NAME'))"
```

**🚀 部署问题**
```bash
# 重新验证 Modal
modal token new

# 检查部署日志
modal app logs poe-bots
```

### **获取帮助**
- 📖 **文档**: 查看 `/examples` 目录
- 🐛 **问题**: 在 GitHub issues 中使用详细错误消息
- 💬 **社区**: 与其他开发者分享经验
- 🔧 **调试模式**: 调查时始终使用 `--debug` 运行

## 📈 性能与可扩展性

- **⚡ 快速响应**: 为快速响应优化
- **📊 可扩展**: 支持多个并发用户
- **🛡️ 可靠**: 为生产工作负载设计
- **💰 成本效益**: 高效的资源使用
- **🔄 自动缩放**: 利用 Modal 的缩放能力

## 🎯 最佳实践

### **机器人开发**
1. **继承 BaseBot + Mixins** 以获得最大功能
2. **使用类型提示** 以获得更好的 IDE 支持和调试
3. **优雅处理错误** 使用 try/catch 和用户友好消息
4. **先在本地测试** 然后再部署
5. **遵循 API 密钥命名约定**

### **生产部署**
1. **设置监控** 通过 Modal 仪表板
2. **使用环境变量** 进行所有配置
3. **与真实 Poe 集成测试** 上线前
4. **监控成本** 并设置计费警报
5. **保持依赖更新** 以确保安全

## 🚀 下一步

1. **🏃‍♂️ 快速开始**: 按照上面的 2 分钟设置
2. **🧪 实验**: 尝试修改包含的示例机器人
3. **🛠️ 构建**: 使用模板创建您的第一个自定义机器人
4. **🚀 部署**: 一键推送到生产环境
5. **📈 扩展**: 根据需要添加更多机器人和功能

## 📋 资源

- **📖 [Poe 文档](https://creator.poe.com/docs)** - 官方 Poe 平台文档
- **🚀 [Modal 文档](https://modal.com/docs)** - 云部署平台
- **⚡ [FastAPI 文档](https://fastapi.tiangolo.com/)** - Web 框架
- **🤖 [FastAPI-Poe](https://github.com/poe-platform/fastapi-poe)** - Poe 机器人框架

## 🤝 贡献

我们欢迎贡献！此框架的设计理念：
- **📚 文档完善**: 每个功能都有示例
- **🧪 测试完备**: 全面的自动化测试确保可靠性
- **🏗️ 架构优良**: 清洁模式使扩展变得容易
- **❤️ 社区导向**: 为开发者而建，由开发者维护

## 📄 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

<div align="center">
  <strong>🎉 准备构建令人惊叹的 Poe 机器人了吗？2 分钟内开始吧！🎉</strong>
  <br><br>
  <em>此框架由社区维护，与 Poe 无官方关联。</em>
</div>
