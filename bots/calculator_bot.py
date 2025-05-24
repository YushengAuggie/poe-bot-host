"""
Calculator Bot - A bot that can perform various mathematical calculations.

This bot demonstrates how to perform calculations, parse user input,
and return formatted results.
"""

import json
import logging
from typing import AsyncGenerator, Union

from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest

from utils.base_bot import BaseBot
from utils.calculators import CalculatorFormatter, SafeExpressionEvaluator, UnitConverter
from utils.mixins import ErrorHandlerMixin, ResponseMixin

logger = logging.getLogger(__name__)


class CalculatorBot(BaseBot, ErrorHandlerMixin, ResponseMixin):
    """
    A bot that performs mathematical calculations.

    Supports:
    - Basic arithmetic (+, -, *, /, ^)
    - Trigonometric functions (sin, cos, tan)
    - Logarithms (log, ln)
    - Unit conversions
    - And more
    """

    bot_name = "CalculatorBot"
    bot_description = (
        "A calculator bot that can perform mathematical calculations and unit conversions."
    )
    version = "1.0.0"

    def __init__(self, **kwargs):
        """Initialize the calculator bot with utilities."""
        super().__init__(**kwargs)
        self.evaluator = SafeExpressionEvaluator()
        self.converter = UnitConverter()
        self.formatter = CalculatorFormatter()

    def _handle_conversion(self, message: str) -> str:
        """Handle unit conversion requests using the UnitConverter utility."""
        success, result = self.converter.parse_conversion_request(message)

        if not success:
            return result  # Error message

        value, from_unit, to_unit = result
        conversion_type = f"{from_unit}_to_{to_unit}"

        try:
            converted_value, unit_label = self.converter.convert(value, conversion_type)
            return self.formatter.format_conversion_response(
                value, from_unit, converted_value, unit_label
            )
        except ValueError as e:
            return str(e)
        except Exception as e:
            logger.error(f"Error in unit conversion: {str(e)}")
            return f"Error: Could not perform conversion. {str(e)}"

    def _get_help_text(self) -> str:
        """Get help text for the calculator bot."""
        help_content = """
I can perform various mathematical calculations. You can:

### Basic Arithmetic
- `2 + 3`
- `10 - 5`
- `7 * 8`
- `20 / 4`
- `2^3` (exponentiation)

### Functions
- `sin(30)`, `cos(45)`, `tan(60)`
- `sqrt(16)`
- `log(100)`, `ln(10)` (natural log)

### Constants
- `pi` - the value of π
- `e` - the value of e

### Unit Conversions
- `convert 32 f to c`
- `convert 10 km to miles`
- `convert 70 kg to lbs`

Type a calculation to begin!
"""
        return self._format_help_response(help_content)

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        """Process the query and generate a response for calculator functions."""

        async def _process_calculation():
            # Extract the query contents
            user_message = self._extract_message(query)
            logger.debug(f"[{self.bot_name}] Received message: {user_message}")

            # Add metadata about the bot if requested
            if user_message.lower().strip() == "bot info":
                metadata = self._get_bot_metadata()
                yield PartialResponse(text=json.dumps(metadata, indent=2))
                return

            message = user_message.strip()

            # Help command
            if message.lower() in ["help", "?", "/help"]:
                yield PartialResponse(text=self._get_help_text())
                return

            # Empty query
            if not message:
                yield PartialResponse(
                    text="Please enter a calculation. Type 'help' for instructions."
                )
                return

            # Unit conversion
            if message.lower().startswith("convert "):
                result = self._handle_conversion(message)
                yield PartialResponse(text=result)
                return

            # Basic calculation
            result = self.evaluator.evaluate(message)
            formatted_response = self.formatter.format_calculation_response(message, result)
            yield PartialResponse(text=formatted_response)

        # Use error handling mixin
        async for response in self.handle_common_errors(query, _process_calculation):
            yield response
