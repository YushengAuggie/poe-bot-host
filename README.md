# 🤖 Poe Bot Host

English | [简体中文](README.zh-CN.md)

A **modern, production-ready** multi-bot hosting framework for the Poe platform. Build, deploy, and manage sophisticated AI bots with clean architecture and enterprise-grade reliability.

```
poe_bots/
├── 🌐 app.py                    # FastAPI application with auto-discovery
├── 🤖 bots/                     # Ready-to-use bot implementations
│   ├── calculator_bot.py        # ➕ Mathematical calculations & unit conversion
│   ├── echo_bot.py              # 🔄 Simple echo with error handling
│   ├── weather_bot.py           # 🌤️ Weather information with API integration
│   ├── chatgpt.py               # 💬 OpenAI GPT integration
│   ├── gemini*.py               # 🧠 Google Gemini AI (multimodal support)
│   └── web_search_bot.py        # 🔍 Web search capabilities
├── 🛠️ utils/                    # Modern, modular architecture
│   ├── mixins.py                # 🎯 Reusable error handling & response patterns
│   ├── auth.py                  # 🔐 Smart API key resolution with fallbacks
│   ├── calculators.py           # 🔢 Secure math evaluation & conversions
│   ├── media/                   # 📁 Media processing framework
│   │   ├── processors.py        # 🖼️ Image/video/audio attachment handling
│   │   ├── validators.py        # ✅ Media type & size validation
│   │   └── converters.py        # 🔄 Format conversion & optimization
│   ├── base_bot.py              # 🏗️ Enhanced base class with auto-discovery
│   └── bot_factory.py           # 🏭 Intelligent bot registration & management
├── 📘 examples/                 # Comprehensive guides & samples
├── 🧪 tests/                    # Comprehensive automated test suite
└── 🚀 Easy deployment           # One-command deployment to Modal/cloud
```

## ✨ Why Choose This Framework?

🎯 **Developer-First Design**: Clean architecture, excellent documentation, comprehensive testing
🔧 **Enterprise Ready**: Robust error handling, secure API key management, production logging
🚀 **Deploy in Minutes**: One command deployment to Modal with auto-scaling
🧩 **Modular & Extensible**: Reusable components, domain-specific utilities, clean abstractions
⚡ **High Performance**: Optimized media processing, intelligent caching, minimal overhead
🛡️ **Security First**: Input sanitization, secure expression evaluation, API key encryption

## 🚀 Quick Start (2 Minutes)

```bash
# 1. Clone and setup
git clone https://github.com/YushengAuggie/poe-bot-host.git
cd poe-bot-host
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run locally (starts all bots automatically)
./run_local.sh

# 3. Test your bots
python scripts/test_bot_cli.py --bot CalculatorBot --message "2 + 2"
python scripts/test_bot_cli.py --bot EchoBot --message "Hello world!"

# 4. Deploy to production (one command!)
modal deploy app.py
```

**That's it!** Your bots are running locally at `http://localhost:8000` 🎉

## 🏗️ Modern Architecture

### 🎯 **Clean Separation of Concerns**

Our refactored architecture follows enterprise patterns for maximum maintainability:

| Component | Purpose | Benefits |
|-----------|---------|----------|
| **🤖 Bots Layer** | Business logic & user interactions | Easy to add new bots, clear responsibilities |
| **🎭 Mixins Layer** | Reusable patterns (errors, responses) | DRY principle, consistent UX across bots |
| **🛠️ Utils Layer** | Domain-specific utilities | Testable, reusable, focused modules |
| **🌐 App Layer** | HTTP routing & bot discovery | Auto-discovery, health checks, monitoring |

### 🧩 **Modular Utilities** (NEW!)

**Media Processing Framework**
```python
from utils.media import MediaProcessor, MediaValidator

processor = MediaProcessor()
attachments = processor.extract_attachments(query)  # Auto-handles images/video/audio
media_parts = processor.prepare_media_parts(attachments)  # Ready for AI models
```

**Smart Authentication**
```python
from utils.auth import APIKeyResolver

resolver = APIKeyResolver("MyBot")
api_key = resolver.resolve()  # Tries multiple naming patterns automatically
```

**Secure Calculations**
```python
from utils.calculators import SafeExpressionEvaluator, UnitConverter

evaluator = SafeExpressionEvaluator()
result = evaluator.evaluate("sin(30) + log(100)")  # Safe from code injection

converter = UnitConverter()
celsius = converter.convert(32, "f_to_c")  # (0.0, "°C")
```

### 🎯 **Enhanced Error Handling**
```python
from utils.mixins import ErrorHandlerMixin, ResponseMixin

class MyBot(BaseBot, ErrorHandlerMixin, ResponseMixin):
    async def get_response(self, query):
        async def _process():
            # Your bot logic here
            yield PartialResponse(text="Hello!")

        # Automatic error handling with retries and user-friendly messages
        async for response in self.handle_common_errors(query, _process):
            yield response
```

## 🤖 Create Your First Bot (30 seconds)

```python
# bots/my_awesome_bot.py
from utils.base_bot import BaseBot
from utils.mixins import ErrorHandlerMixin, ResponseMixin
from fastapi_poe.types import PartialResponse, QueryRequest

class MyAwesomeBot(BaseBot, ErrorHandlerMixin, ResponseMixin):
    bot_name = "MyAwesomeBot"
    bot_description = "Does something amazing!"

    async def get_response(self, query: QueryRequest):
        async def _process():
            user_message = self._extract_message(query)

            if user_message.lower() == "help":
                help_text = "I can help you with amazing things!\nTry: 'hello', 'joke', 'advice'"
                yield PartialResponse(text=self._format_help_response(help_text))
                return

            if "hello" in user_message.lower():
                yield PartialResponse(text="👋 Hello! I'm your awesome bot!")
            elif "joke" in user_message.lower():
                yield PartialResponse(text="🤖 Why did the bot cross the road? To get to the other byte!")
            else:
                yield PartialResponse(text=f"✨ You said: '{user_message}' - that's awesome!")

        async for response in self.handle_common_errors(query, _process):
            yield response
```

**Features you get for free:**
- ✅ Automatic error handling with user-friendly messages
- ✅ Consistent response formatting with emoji support
- ✅ API key management (if needed)
- ✅ Help command formatting
- ✅ Logging and debugging
- ✅ Auto-discovery by the framework

## 🔑 API Key Management Made Simple

### **Two Types of Keys**
1. **🔌 Service API Keys** (OpenAI, Google, etc.) - for external services
2. **🤖 Bot Access Keys** - unique Poe keys for each bot (get from Poe dashboard)

### **Setup (One-Time)**
Create `.env` file in your project root:
```bash
# Service API keys (optional - only if your bots need them)
OPENAI_API_KEY=sk-...your-openai-key...
GOOGLE_API_KEY=AIza...your-google-key...

# Bot access keys from Poe (required for deployment)
ECHO_BOT_ACCESS_KEY=psk_...from-poe-dashboard...
CALCULATOR_BOT_ACCESS_KEY=psk_...from-poe-dashboard...
MYAWESOME_BOT_ACCESS_KEY=psk_...from-poe-dashboard...
```

### **Flexible Naming**
The framework automatically finds your keys using multiple patterns:
```bash
# All of these work for "WeatherBot":
WEATHER_BOT_ACCESS_KEY=psk_...     # ✅ Recommended
WEATHERBOT_ACCESS_KEY=psk_...      # ✅ Also works
WeatherBot_ACCESS_KEY=psk_...      # ✅ Also works
```

### **Get Bot Access Keys**
1. Go to https://poe.com/edit_bot?bot=YOUR_BOT_NAME
2. Copy the access key from the page
3. Add to your `.env` file

**That's it!** The framework handles everything else automatically.

## 🚀 Deployment (Production Ready)

### **Deploy to Modal (Recommended)**
```bash
# One command deployment
modal deploy app.py

# Output example:
# ✅ Modal app deployed!
# 🌐 https://yourname--poe-bots-fastapi-app.modal.run
```

### **Connect to Poe**
1. **Create Bot** on [Poe Creator Portal](https://creator.poe.com/)
2. **Configure Server Settings**:
   - **Endpoint**: `https://yourname--poe-bots-fastapi-app.modal.run/mybotname`
   - **Protocol**: Poe Protocol
   - **Protection**: No protection (or configure API key)
3. **Test**: Send a message to your bot!

### **Key Features**
- ✅ **Easy deployment**: One-command deployment to Modal
- ✅ **Health monitoring**: Built-in health check endpoints
- ✅ **Error handling**: Comprehensive error logging
- ✅ **Flexible hosting**: Deploy locally or to the cloud

## 🛠️ Development Workflow

### **Local Development**
```bash
# Start development server with auto-reload
./run_local.sh --debug --reload

# Test specific bot
python scripts/test_bot_cli.py --bot YourBot --message "test message"

# Check all bots are working
python scripts/test_bot_cli.py --health
```

### **Testing & Quality**
```bash
# Run all 257 automated tests
pytest tests/ -v

# Check code formatting
black . && ruff check .

# Type checking
pyright .
```

### **Continuous Integration**
- ✅ **Pre-commit hooks**: Automatic code formatting and linting
- ✅ **GitHub Actions**: Automated testing on every push
- ✅ **Quality gates**: Type checking, security scans, test coverage

## 📚 Advanced Features

### **🖼️ Multimodal Support (Images/Video/Audio)**
```python
# Built-in support for media attachments
attachments = self.media_processor.extract_attachments(query)
if attachments:
    for attachment in attachments:
        # Process images, videos, or audio files
        processed = self.media_processor.process_attachment(attachment)
```

### **🔒 Secure Expression Evaluation**
```python
# Safe mathematical calculations (no code injection)
from utils.calculators import SafeExpressionEvaluator

evaluator = SafeExpressionEvaluator()
result = evaluator.evaluate("2 + 2 * sin(30)")  # ✅ Safe
result = evaluator.evaluate("import os; os.system('rm -rf /')")  # ❌ Blocked
```

### **🎯 Response Formatting**
```python
# Consistent, beautiful responses
self._format_help_response("Your help text")  # 🤖 BotName Help
self._format_error_response("Something went wrong")  # ❌ Error: Something went wrong
self._format_success_response("Task completed")  # ✅ Task completed
```

### **📊 Built-in Analytics**
- Bot usage metrics via `/health` endpoint
- Performance monitoring
- Error rate tracking
- Response time analysis

## 🧪 Example Bots

The framework includes several ready-to-use bots:

- **Basic bots**: Echo, Calculator, Weather
- **AI-powered bots**: ChatGPT, Gemini (with multimodal support)
- **Utility bots**: Web Search, File Analyzer, Function Calling
- **Template bot**: Use as starting point for new bots

Explore the `bots/` directory to see all available implementations.

## 🆘 Troubleshooting

### **Common Issues**

**🔍 Bot not found**
```bash
# Check if bot is discovered
python scripts/test_bot_cli.py --list-bots

# Verify bot class inherits from BaseBot
class MyBot(BaseBot):  # ✅ Correct
```

**🔐 Authentication errors**
```bash
# Check API keys are loaded
python -c "import os; print([k for k in os.environ if 'API_KEY' in k])"

# Verify .env file exists and is loaded
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('YOUR_KEY_NAME'))"
```

**🚀 Deployment issues**
```bash
# Re-authenticate with Modal
modal token new

# Check deployment logs
modal app logs poe-bots
```

### **Getting Help**
- 📖 **Documentation**: Check the `/examples` directory
- 🐛 **Issues**: Use detailed error messages in GitHub issues
- 💬 **Community**: Share experiences with other developers
- 🔧 **Debug mode**: Always run with `--debug` when investigating

## 📈 Performance & Scalability

- **⚡ Fast Response**: Optimized for quick responses
- **📊 Scalable**: Built to handle multiple concurrent users
- **🛡️ Reliable**: Designed for production workloads
- **💰 Cost-effective**: Efficient resource usage
- **🔄 Auto-scaling**: Leverages Modal's scaling capabilities

## 🎯 Best Practices

### **Bot Development**
1. **Inherit from BaseBot + Mixins** for maximum functionality
2. **Use type hints** for better IDE support and debugging
3. **Handle errors gracefully** with try/catch and user-friendly messages
4. **Test locally first** before deploying
5. **Follow naming conventions** for API keys

### **Production Deployment**
1. **Set up monitoring** via Modal dashboard
2. **Use environment variables** for all configuration
3. **Test with real Poe integration** before going live
4. **Monitor costs** and set up billing alerts
5. **Keep dependencies updated** for security

## 🚀 Next Steps

1. **🏃‍♂️ Quick Start**: Follow the 2-minute setup above
2. **🧪 Experiment**: Try modifying the included example bots
3. **🛠️ Build**: Create your first custom bot using the template
4. **🚀 Deploy**: Push to production with one command
5. **📈 Scale**: Add more bots and features as needed

## 📋 Resources

- **📖 [Poe Documentation](https://creator.poe.com/docs)** - Official Poe platform docs
- **🚀 [Modal Documentation](https://modal.com/docs)** - Cloud deployment platform
- **⚡ [FastAPI Documentation](https://fastapi.tiangolo.com/)** - Web framework
- **🤖 [FastAPI-Poe](https://github.com/poe-platform/fastapi-poe)** - Poe bot framework

## 🤝 Contributing

We welcome contributions! This framework is designed to be:
- **📚 Well-documented**: Every feature has examples
- **🧪 Well-tested**: Comprehensive automated tests ensure reliability
- **🏗️ Well-architected**: Clean patterns make it easy to extend
- **❤️ Community-focused**: Built for developers, by developers

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>🎉 Ready to build amazing Poe bots? Get started in 2 minutes! 🎉</strong>
  <br><br>
  <em>This framework is community-maintained and not officially affiliated with Poe.</em>
</div>
