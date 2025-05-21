"""
Calculator Bot - A bot that can perform various mathematical calculations.

This bot demonstrates how to perform calculations, parse user input,
and return formatted results.
"""

import json
import logging
import math
import re
from typing import AsyncGenerator, Tuple, Union

from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest

from utils.base_bot import BaseBot

logger = logging.getLogger(__name__)


class CalculatorBot(BaseBot):
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

    # Constants for calculations
    PI = math.pi
    E = math.e

    # Available operations
    OPERATIONS = {
        "basic": ["add", "subtract", "multiply", "divide", "power", "sqrt", "factorial"],
        "trig": ["sin", "cos", "tan", "asin", "acos", "atan"],
        "logarithmic": ["log", "ln", "log10"],
        "conversion": ["c_to_f", "f_to_c", "km_to_miles", "miles_to_km", "kg_to_lbs", "lbs_to_kg"],
    }

    def _parse_expression(self, expression: str) -> Union[float, str]:
        """
        Parse and evaluate a mathematical expression.

        Args:
            expression: The mathematical expression to evaluate

        Returns:
            The result of the evaluation
        """
        # Remove whitespace and convert to lowercase
        expression = expression.strip().lower()

        # Handle specific functions
        if expression.startswith(("sin(", "cos(", "tan(", "log(", "ln(", "sqrt(")):
            return self._evaluate_function(expression)

        # Basic arithmetic evaluation (careful with security!)
        try:
            # Replace ^ with ** for exponentiation
            expression = expression.replace("^", "**")

            # Replace constants with their values
            expression = expression.replace("pi", str(self.PI))
            expression = expression.replace("e", str(self.E))

            # Validate the expression (only allow safe characters)
            if not re.match(r"^[0-9+\-*/().\s**]+$", expression):
                return "Invalid expression. Only basic arithmetic operations are allowed."

            # Evaluate the expression
            # Note: eval() can be dangerous, but we've sanitized the input
            result = eval(expression)
            return result
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            logger.error(f"Error evaluating expression '{expression}': {str(e)}")
            return f"Error: Could not evaluate expression. {str(e)}"

    def _evaluate_function(self, expression: str) -> Union[float, str]:
        """Evaluate a function expression like sin(30), log(100), etc."""
        try:
            # Extract function name and argument
            match = re.match(r"([a-z]+)\(([^)]+)\)", expression)
            if not match:
                return "Invalid function format. Use function(value)."

            func_name, arg_str = match.groups()

            # Evaluate the argument first (it might be an expression)
            arg = self._parse_expression(arg_str)
            if isinstance(arg, str):
                return arg  # If arg evaluation returned an error

            # Handle different functions
            if func_name == "sin":
                return math.sin(arg)
            elif func_name == "cos":
                return math.cos(arg)
            elif func_name == "tan":
                return math.tan(arg)
            elif func_name == "asin":
                return math.asin(arg)
            elif func_name == "acos":
                return math.acos(arg)
            elif func_name == "atan":
                return math.atan(arg)
            elif func_name == "log" or func_name == "ln":
                return math.log(arg)
            elif func_name == "log10":
                return math.log10(arg)
            elif func_name == "sqrt":
                return math.sqrt(arg)
            else:
                return f"Unknown function: {func_name}"

        except Exception as e:
            logger.error(f"Error evaluating function '{expression}': {str(e)}")
            return f"Error: Could not evaluate function. {str(e)}"

    def _convert_units(self, value: float, conversion_type: str) -> Tuple[float, str]:
        """
        Convert between different units.

        Args:
            value: The value to convert
            conversion_type: The type of conversion to perform

        Returns:
            Tuple of (converted value, unit string)
        """
        if conversion_type == "c_to_f":
            return (value * 9 / 5) + 32, "°F"
        elif conversion_type == "f_to_c":
            return (value - 32) * 5 / 9, "°C"
        elif conversion_type == "km_to_miles":
            return value * 0.621371, "miles"
        elif conversion_type == "miles_to_km":
            return value * 1.60934, "km"
        elif conversion_type == "kg_to_lbs":
            return value * 2.20462, "lbs"
        elif conversion_type == "lbs_to_kg":
            return value * 0.453592, "kg"
        else:
            raise ValueError(f"Unknown conversion type: {conversion_type}")

    def _handle_conversion(self, message: str) -> str:
        """Handle unit conversion requests."""
        try:
            # Parse conversion request
            # Format: "convert 100 c to f", "convert 10 km to miles", etc.
            match = re.match(
                r"convert\s+(\d+(?:\.\d+)?)\s+([a-z]+)\s+to\s+([a-z]+)", message.lower()
            )
            if not match:
                return "Invalid conversion format. Use 'convert {value} {from_unit} to {to_unit}'."

            value_str, from_unit, to_unit = match.groups()
            value = float(value_str)

            # Determine conversion type
            conversion_type = f"{from_unit}_to_{to_unit}"
            if conversion_type not in self.OPERATIONS["conversion"]:
                return f"Unsupported conversion: {from_unit} to {to_unit}"

            # Perform conversion
            result, unit = self._convert_units(value, conversion_type)
            return f"{value} {from_unit} = {result:.4f} {unit}"

        except Exception as e:
            logger.error(f"Error in unit conversion: {str(e)}")
            return f"Error: Could not perform conversion. {str(e)}"

    def _get_help_text(self) -> str:
        """Get help text for the calculator bot."""
        return """
## 🧮 Calculator Bot

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

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        """Process the query and generate a response for calculator functions."""
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
            try:
                result = self._parse_expression(message)

                # Format the result
                if isinstance(result, float):
                    # Handle very small numbers near zero
                    if abs(result) < 1e-10:
                        result = 0

                    # Format with appropriate precision
                    if isinstance(result, float) and result.is_integer():
                        formatted_result = str(int(result))
                    else:
                        formatted_result = f"{result:.6f}".rstrip("0").rstrip(".")
                else:
                    formatted_result = str(result)

                yield PartialResponse(text=f"```\n{message} = {formatted_result}\n```")

            except Exception as e:
                yield PartialResponse(text=f"Calculation error: {str(e)}")
                return

        except Exception:
            # Let the parent class handle errors
            async for resp in super().get_response(query):
                yield resp
