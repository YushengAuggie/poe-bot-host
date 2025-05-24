"""
Calculator utilities extracted from CalculatorBot.

This module provides safe mathematical expression evaluation and
unit conversion functionality.
"""

import logging
import math
import re
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)


class SafeExpressionEvaluator:
    """Safely evaluates mathematical expressions."""

    # Mathematical constants
    CONSTANTS = {"pi": math.pi, "e": math.e}

    # Allowed characters in expressions
    SAFE_CHARS_PATTERN = re.compile(r"^[0-9+\-*/().\s**]+$")

    def __init__(self):
        """Initialize the evaluator."""
        self.allowed_functions = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "log": math.log,
            "ln": math.log,
            "log10": math.log10,
            "sqrt": math.sqrt,
        }

    def evaluate(self, expression: str) -> Union[float, str]:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: The mathematical expression to evaluate

        Returns:
            The result as a float, or an error message string
        """
        try:
            expression = expression.strip().lower()

            # Handle function expressions
            if self._is_function_expression(expression):
                return self._evaluate_function(expression)

            # Handle basic arithmetic
            return self._evaluate_arithmetic(expression)

        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            logger.error(f"Error evaluating expression '{expression}': {str(e)}")
            return f"Error: Could not evaluate expression. {str(e)}"

    def _is_function_expression(self, expression: str) -> bool:
        """Check if expression is a function call."""
        return any(func in expression for func in self.allowed_functions.keys())

    def _evaluate_function(self, expression: str) -> Union[float, str]:
        """Evaluate a function expression like sin(30), log(100), etc."""
        try:
            # Extract function name and argument
            match = re.match(r"([a-z]+)\(([^)]+)\)", expression)
            if not match:
                return "Invalid function format. Use function(value)."

            func_name, arg_str = match.groups()

            # Check if function is allowed
            if func_name not in self.allowed_functions:
                return f"Unknown function: {func_name}"

            # Evaluate the argument (might be an expression)
            arg_result = self.evaluate(arg_str)
            if isinstance(arg_result, str):
                return arg_result  # Error in argument evaluation

            # Apply the function
            func = self.allowed_functions[func_name]
            return func(arg_result)

        except Exception as e:
            logger.error(f"Error evaluating function '{expression}': {str(e)}")
            return f"Error: Could not evaluate function. {str(e)}"

    def _evaluate_arithmetic(self, expression: str) -> Union[float, str]:
        """Evaluate basic arithmetic expressions."""
        # Replace ^ with ** for exponentiation
        expression = expression.replace("^", "**")

        # Replace constants
        for const_name, const_value in self.CONSTANTS.items():
            expression = expression.replace(const_name, str(const_value))

        # Validate expression (only allow safe characters)
        if not self.SAFE_CHARS_PATTERN.match(expression):
            return "Invalid expression. Only basic arithmetic operations are allowed."

        # Evaluate using eval (input is sanitized)
        try:
            result = eval(expression)
            return float(result)
        except Exception as e:
            return f"Error: {str(e)}"


class UnitConverter:
    """Handles unit conversions between different measurement systems."""

    # Conversion functions
    CONVERSIONS = {
        "c_to_f": lambda x: (x * 9 / 5) + 32,
        "f_to_c": lambda x: (x - 32) * 5 / 9,
        "km_to_miles": lambda x: x * 0.621371,
        "miles_to_km": lambda x: x * 1.60934,
        "kg_to_lbs": lambda x: x * 2.20462,
        "lbs_to_kg": lambda x: x * 0.453592,
    }

    # Unit labels for display
    UNIT_LABELS = {
        "c_to_f": "°F",
        "f_to_c": "°C",
        "km_to_miles": "miles",
        "miles_to_km": "km",
        "kg_to_lbs": "lbs",
        "lbs_to_kg": "kg",
    }

    def convert(self, value: float, conversion_type: str) -> Tuple[float, str]:
        """
        Convert between different units.

        Args:
            value: The value to convert
            conversion_type: The type of conversion (e.g., "c_to_f")

        Returns:
            Tuple of (converted_value, unit_label)

        Raises:
            ValueError: If conversion type is not supported
        """
        if conversion_type not in self.CONVERSIONS:
            raise ValueError(f"Unknown conversion type: {conversion_type}")

        converter = self.CONVERSIONS[conversion_type]
        unit_label = self.UNIT_LABELS[conversion_type]

        return converter(value), unit_label

    def parse_conversion_request(
        self, message: str
    ) -> Tuple[bool, Union[str, Tuple[float, str, str]]]:
        """
        Parse a conversion request from user message.

        Args:
            message: User message like "convert 100 c to f"

        Returns:
            Tuple of (success, result_or_error)
            If success: (True, (value, from_unit, to_unit))
            If error: (False, error_message)
        """
        try:
            # Parse conversion request
            match = re.match(
                r"convert\s+(\d+(?:\.\d+)?)\s+([a-z]+)\s+to\s+([a-z]+)", message.lower()
            )

            if not match:
                return (
                    False,
                    "Invalid conversion format. Use 'convert {value} {from_unit} to {to_unit}'.",
                )

            value_str, from_unit, to_unit = match.groups()
            value = float(value_str)

            # Check if conversion is supported
            conversion_type = f"{from_unit}_to_{to_unit}"
            if conversion_type not in self.CONVERSIONS:
                return False, f"Unsupported conversion: {from_unit} to {to_unit}"

            return True, (value, from_unit, to_unit)

        except ValueError:
            return False, "Invalid number format"
        except Exception as e:
            return False, f"Error parsing conversion: {str(e)}"

    def get_supported_conversions(self) -> List[str]:
        """Get list of supported conversion types."""
        return list(self.CONVERSIONS.keys())


class CalculatorFormatter:
    """Formats calculator results for display."""

    @staticmethod
    def format_result(result: Union[float, str]) -> str:
        """
        Format a calculation result for display.

        Args:
            result: The result to format

        Returns:
            Formatted result string
        """
        if isinstance(result, str):
            return result  # Already formatted (error message)

        # Handle very small numbers near zero
        if abs(result) < 1e-10:
            result = 0

        # Format with appropriate precision
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        else:
            return f"{result:.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def format_calculation_response(expression: str, result: Union[float, str]) -> str:
        """
        Format a complete calculation response.

        Args:
            expression: The original expression
            result: The calculated result

        Returns:
            Formatted response with code block
        """
        formatted_result = CalculatorFormatter.format_result(result)
        return f"```\n{expression} = {formatted_result}\n```"

    @staticmethod
    def format_conversion_response(
        value: float, from_unit: str, converted_value: float, to_unit: str
    ) -> str:
        """
        Format a unit conversion response.

        Args:
            value: Original value
            from_unit: Original unit
            converted_value: Converted value
            to_unit: Target unit

        Returns:
            Formatted conversion response
        """
        return f"{value} {from_unit} = {converted_value:.4f} {to_unit}"
