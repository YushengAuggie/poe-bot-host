# Adding the Weather Bot to Your Framework

This guide shows how to add the example weather bot to your Poe Bots Framework.

## Step 1: Copy the Weather Bot Implementation

Copy the `weather_bot.py` file from the examples directory to your bots directory:

```bash
cp examples/weather_bot.py bots/weather_bot.py
```

## Step 2: Add the Import to `bots/__init__.py`

Edit the `bots/__init__.py` file to include the WeatherBot:

```python
"""
Package containing all the Poe bot implementations.

All bot classes are automatically loaded if they:
1. Inherit from BaseBot or PoeBot
2. Are defined in this package or submodules

To create a new bot:
1. Create a new file in this directory
2. Define a class that inherits from BaseBot
3. Override get_response to implement the bot's behavior

The bot will be automatically discovered and loaded when the app starts.
"""

# Import all bots for easy access
from .echo_bot import EchoBot
from .reverse_bot import ReverseBot
from .uppercase_bot import UppercaseBot
from .weather_bot import WeatherBot  # Add this line

# Add other bots here as you create them
# from .my_other_bot import MyOtherBot

# Export the bot classes
__all__ = ["EchoBot", "ReverseBot", "UppercaseBot", "WeatherBot"]  # Update this line
```

## Step 3: Test the Weather Bot Locally

1. Start the local server:
   ```bash
   ./run_local.sh
   ```

2. In a new terminal, test the weather bot:
   ```bash
   python test_bot.py --bot WeatherBot --message "Weather in London"
   ```

3. You should see a weather report for London:
   ```
   Checking weather for London...

   🌡️ **Weather for London**

   **Current Conditions:** Cloudy
   **Temperature:** 15°C / 59°F
   **Humidity:** 78%
   **Wind Speed:** 12 km/h

   **3-Day Forecast:**
   - Monday: Rainy, 10°C to 16°C
   - Tuesday: Partly Cloudy, 9°C to 14°C
   - Wednesday: Cloudy, 11°C to 18°C

   *Note: This is simulated weather data for demonstration purposes.*
   ```

## Step 4: Deploy to Modal

Deploy your updated bot collection:

```bash
modal deploy app.py
```

## Step 5: Create a Weather Bot on Poe

1. Go to [creator.poe.com](https://creator.poe.com/)
2. Click "Create Bot"
3. Fill in the details:
   - **Name**: WeatherBot
   - **Description**: Get weather information for any location
4. Configure server settings:
   - **Server Endpoint**: `https://yourusername--poe-bots-fastapi-app.modal.run/weatherbot`
   - **API Protocol**: Poe Protocol
5. Add sample messages:
   - "Weather in New York"
   - "What's the weather like in Tokyo?"
   - "London weather"
6. Save and test your bot!
