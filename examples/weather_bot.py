"""
Example of a more complex bot that simulates a weather service.

This bot demonstrates how to create a more sophisticated bot with:
- Custom message parsing
- Simulated API calls
- Error handling
- Structured responses
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, Optional, Union

from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest

from utils.base_bot import BaseBot, BotError

# Get logger
logger = logging.getLogger(__name__)


class WeatherBot(BaseBot):
    """A bot that provides simulated weather information."""

    bot_name = "WeatherBot"
    bot_description = "Get weather information for any location (simulated)"
    version = "1.0.0"

    # List of cities for random weather generation
    CITIES = [
        "New York",
        "London",
        "Tokyo",
        "Paris",
        "Sydney",
        "Berlin",
        "Toronto",
        "Singapore",
        "Dubai",
        "Mumbai",
        "São Paulo",
    ]

    # Weather conditions for simulation
    WEATHER_CONDITIONS = [
        "Sunny",
        "Partly Cloudy",
        "Cloudy",
        "Rainy",
        "Thunderstorms",
        "Snowy",
        "Foggy",
        "Windy",
        "Clear",
    ]

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        """Process the query and generate a weather report response."""
        try:
            # Extract the query contents
            user_message = self._extract_message(query)

            # Log the extracted message
            logger.debug(f"[{self.bot_name}] Received message: {user_message}")

            # Add metadata about the bot if requested
            if user_message.lower().strip() == "bot info":
                metadata = self._get_bot_metadata()
                yield PartialResponse(text=json.dumps(metadata, indent=2))
                return

            # Extract location from message
            location = self._parse_location(user_message)

            if not location:
                yield PartialResponse(
                    text="Please provide a city name for the weather forecast. For example: 'Weather in London'"
                )
                return

            # Simulate API call delay
            yield PartialResponse(text=f"Checking weather for {location}...\n\n")
            await asyncio.sleep(1)  # Simulate API latency

            # Get weather data (simulated)
            try:
                weather_data = await self._get_weather_data(location)

                # Format and return the weather report
                report = self._format_weather_report(location, weather_data)
                yield PartialResponse(text=report)

            except BotError as e:
                # Handle expected errors
                yield PartialResponse(
                    text=f"Sorry, I couldn't get the weather for {location}: {str(e)}"
                )

        except Exception:
            # Let the parent class handle errors
            async for resp in super().get_response(query):
                yield resp

    def _parse_location(self, message: str) -> Optional[str]:
        """Extract location from the user's message."""
        message = message.lower().strip()

        # Check for specific location patterns
        if "weather in " in message:
            return message.split("weather in ")[1].strip().title()
        elif "weather for " in message:
            return message.split("weather for ")[1].strip().title()
        elif "weather " in message:
            return message.split("weather ")[1].strip().title()
        elif message in [city.lower() for city in self.CITIES]:
            return message.title()

        # Check if the message is just a city name
        words = message.split()
        if len(words) == 1 and len(words[0]) > 2:
            return words[0].title()

        return None

    async def _get_weather_data(self, location: str) -> Dict[str, Any]:
        """Simulate getting weather data for a location."""
        # If city is known, generate consistent simulated data
        if location in self.CITIES:
            # Use deterministic randomness based on city name
            seed = sum(ord(c) for c in location) + datetime.now().day
            random.seed(seed)

            temp_c = random.randint(5, 35)
            condition = random.choice(self.WEATHER_CONDITIONS)
            humidity = random.randint(30, 90)
            wind_speed = random.randint(0, 30)

            # Generate forecast for next 3 days
            forecast = []
            for i in range(1, 4):
                day = (datetime.now() + timedelta(days=i)).strftime("%A")
                forecast.append(
                    {
                        "day": day,
                        "condition": random.choice(self.WEATHER_CONDITIONS),
                        "temp_high": temp_c + random.randint(-3, 3),
                        "temp_low": temp_c + random.randint(-8, -2),
                    }
                )

            return {
                "temp_c": temp_c,
                "temp_f": round(temp_c * 9 / 5 + 32),
                "condition": condition,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "forecast": forecast,
            }
        else:
            # Simulate an API error for unknown locations
            if random.random() < 0.3:  # 30% chance of error
                raise BotError(f"Weather service unavailable for {location}")

            # Generate random weather for unknown location
            temp_c = random.randint(5, 35)
            return {
                "temp_c": temp_c,
                "temp_f": round(temp_c * 9 / 5 + 32),
                "condition": random.choice(self.WEATHER_CONDITIONS),
                "humidity": random.randint(30, 90),
                "wind_speed": random.randint(0, 30),
                "forecast": [],  # No forecast for unknown locations
            }

    def _format_weather_report(self, location: str, data: Dict[str, Any]) -> str:
        """Format weather data into a readable report."""
        report = [
            f"🌡️ **Weather for {location}**",
            "",
            f"**Current Conditions:** {data['condition']}",
            f"**Temperature:** {data['temp_c']}°C / {data['temp_f']}°F",
            f"**Humidity:** {data['humidity']}%",
            f"**Wind Speed:** {data['wind_speed']} km/h",
            "",
        ]

        # Add forecast if available
        if data["forecast"]:
            report.append("**3-Day Forecast:**")
            for day in data["forecast"]:
                report.append(
                    f"- {day['day']}: {day['condition']}, {day['temp_low']}°C to {day['temp_high']}°C"
                )

        # Add disclaimer
        report.append("")
        report.append("*Note: This is simulated weather data for demonstration purposes.*")

        return "\n".join(report)


# For standalone testing
if __name__ == "__main__":
    # Use asyncio already imported at the top

    async def test_bot():
        bot = WeatherBot()
        from fastapi_poe.types import ProtocolMessage

        # Use the new format for query with proper message type
        message = ProtocolMessage(role="user", content="Weather in London")
        query = QueryRequest(
            query=[message],
            user_id="test_user",
            conversation_id="test_conversation",
            message_id="test_message",
            version="1.0",
            type="query",
        )

        print("Testing WeatherBot with 'Weather in London':")
        async for response in bot.get_response(query):
            print(response.text)

    asyncio.run(test_bot())
