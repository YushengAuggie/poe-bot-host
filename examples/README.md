# Poe Bots Framework Examples

This directory contains example bots and guides to help you get started with the Poe Bots Framework. These examples have been specially designed to demonstrate different aspects of bot development.

## Example Bots

### StandaloneEchoBot (`standalone_echobot.py`)

A simple example of a standalone bot that can be deployed independently of the main framework. This demonstrates how to create a single-bot deployment.

To deploy:
```bash
modal deploy examples/standalone_echobot.py
```

### WeatherBot (`weather_bot.py`)

A more sophisticated example bot that simulates a weather service. This demonstrates:
- Custom message parsing
- Simulated API calls
- Error handling
- Structured responses

See `add_weather_bot.md` for a guide on adding this bot to your framework.

## Example Guides

### Add Weather Bot (`add_weather_bot.md`)

A step-by-step guide showing how to add the WeatherBot to your framework and deploy it.

## Creating Your Own Examples

Feel free to create your own examples in this directory. Here are some ideas:

1. **Translation Bot**: Simulate translating text between languages
2. **Quiz Bot**: Create a simple quiz game
3. **Reminder Bot**: Simulate setting and recalling reminders
4. **Recipe Bot**: Provide recipes based on ingredient searches
5. **News Bot**: Simulate retrieving and summarizing news articles

## How to Run Examples

You can run any example bot directly:

```bash
python examples/weather_bot.py
```

Or test it through the test script after adding it to your bots directory:

```bash
python test_bot.py --bot WeatherBot --message "Weather in London"
```
