"""
Function Calling Bot - A bot that demonstrates function calling with the Poe API.

This is a demonstration of the Poe API function calling capabilities,
allowing bots to make calls to predefined functions based on user requests.
"""

import logging
import json
import math
import re
import os
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, List, Optional, Callable
from fastapi_poe.types import PartialResponse, QueryRequest, ProtocolMessage, MetaResponse
from utils.base_bot import BaseBot, BotError, BotErrorNoRetry

logger = logging.getLogger(__name__)

class FunctionCallingBot(BaseBot):
    """
    A bot that demonstrates function calling with the Poe API.
    
    This bot can:
    - Calculate simple math expressions
    - Convert units
    - Get the current time
    - Generate random numbers
    """
    
    bot_name = "FunctionCallingBot"
    bot_description = "A bot that demonstrates function calling with the Poe API."
    version = "1.0.0"
    
    def __init__(self, **kwargs):
        """Initialize the FunctionCallingBot."""
        super().__init__(**kwargs)
        
        # Define available functions
        self.functions = {
            "calculate": {
                "name": "calculate",
                "description": "Calculate the result of a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            },
            "convert_units": {
                "name": "convert_units",
                "description": "Convert a value from one unit to another",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "number",
                            "description": "The value to convert"
                        },
                        "from_unit": {
                            "type": "string",
                            "description": "The unit to convert from"
                        },
                        "to_unit": {
                            "type": "string",
                            "description": "The unit to convert to"
                        }
                    },
                    "required": ["value", "from_unit", "to_unit"]
                }
            },
            "get_current_time": {
                "name": "get_current_time",
                "description": "Get the current time in a specified timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone to get the current time for (e.g., UTC, local)"
                        }
                    },
                    "required": ["timezone"]
                }
            },
            "generate_random_number": {
                "name": "generate_random_number",
                "description": "Generate a random number between min and max (inclusive)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min": {
                            "type": "integer",
                            "description": "The minimum value"
                        },
                        "max": {
                            "type": "integer",
                            "description": "The maximum value"
                        }
                    },
                    "required": ["min", "max"]
                }
            }
        }
        
        # Define function implementations
        self.function_implementations = {
            "calculate": self._calculate,
            "convert_units": self._convert_units,
            "get_current_time": self._get_current_time,
            "generate_random_number": self._generate_random_number
        }
    
    def _calculate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the result of a mathematical expression.
        
        Args:
            params: Dictionary with the expression to evaluate
            
        Returns:
            Dictionary with the calculation result
        """
        expression = params.get("expression", "")
        
        try:
            # Simple sanitization - only allow basic arithmetic operations
            # This is a basic implementation - in production, you'd want more safety
            if not re.match(r'^[0-9+\-*/().\s]*$', expression):
                return {"error": "Invalid expression. Only basic arithmetic operations are allowed."}
            
            # Evaluate the expression
            result = eval(expression)
            return {"result": result}
        except Exception as e:
            return {"error": f"Error evaluating expression: {str(e)}"}
    
    def _convert_units(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a value from one unit to another.
        
        Args:
            params: Dictionary with value, from_unit, and to_unit
            
        Returns:
            Dictionary with the conversion result
        """
        value = params.get("value", 0)
        from_unit = params.get("from_unit", "").lower()
        to_unit = params.get("to_unit", "").lower()
        
        # Define unit conversions
        conversions = {
            "km_to_miles": lambda x: x * 0.621371,
            "miles_to_km": lambda x: x * 1.60934,
            "kg_to_lbs": lambda x: x * 2.20462,
            "lbs_to_kg": lambda x: x * 0.453592,
            "c_to_f": lambda x: (x * 9/5) + 32,
            "f_to_c": lambda x: (x - 32) * 5/9,
            "m_to_ft": lambda x: x * 3.28084,
            "ft_to_m": lambda x: x * 0.3048,
            "l_to_gal": lambda x: x * 0.264172,
            "gal_to_l": lambda x: x * 3.78541
        }
        
        # Check if conversion is supported
        conversion_key = f"{from_unit}_to_{to_unit}"
        if conversion_key not in conversions:
            return {"error": f"Unsupported conversion from {from_unit} to {to_unit}"}
        
        # Perform conversion
        converted_value = conversions[conversion_key](value)
        return {
            "from_value": value,
            "from_unit": from_unit,
            "to_value": converted_value,
            "to_unit": to_unit
        }
    
    def _get_current_time(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the current time in a specified timezone.
        
        Args:
            params: Dictionary with timezone
            
        Returns:
            Dictionary with the current time
        """
        timezone = params.get("timezone", "").lower()
        
        # Get current time
        now = datetime.now()
        
        # Format time based on timezone (simplified implementation)
        if timezone == "utc":
            return {
                "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "timezone": "UTC"
            }
        else:
            return {
                "time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "timezone": "Local"
            }
    
    def _generate_random_number(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a random number between min and max (inclusive).
        
        Args:
            params: Dictionary with min and max values
            
        Returns:
            Dictionary with the random number
        """
        import random
        
        min_val = params.get("min", 1)
        max_val = params.get("max", 100)
        
        # Validate input
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        
        # Generate random number
        random_number = random.randint(min_val, max_val)
        return {
            "random_number": random_number,
            "min": min_val,
            "max": max_val
        }
    
    def _call_function(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a function by name with parameters.
        
        Args:
            function_name: Name of the function to call
            parameters: Function parameters
            
        Returns:
            Result of the function call
        """
        if function_name not in self.function_implementations:
            return {"error": f"Unknown function: {function_name}"}
        
        try:
            function = self.function_implementations[function_name]
            return function(parameters)
        except Exception as e:
            logger.error(f"Error calling function {function_name}: {str(e)}")
            return {"error": f"Error calling function {function_name}: {str(e)}"}
    
    async def _process_message(self, message: str, query: QueryRequest) -> AsyncGenerator[PartialResponse, None]:
        """
        Process the user's message and handle function calling.
        """
        message = message.strip()
        
        # Help command
        if message.lower() in ["help", "?", "/help"]:
            yield PartialResponse(text="""
## ⚙️ Function Calling Bot

I can demonstrate the Poe API function calling capabilities. I can:

- **Calculate** mathematical expressions
  - `Calculate 2 + 2`
  - `What's the square root of 16?`

- **Convert Units**
  - `Convert 10 kilometers to miles`
  - `How many pounds is 5 kilograms?`

- **Get Current Time**
  - `What time is it?`
  - `Tell me the UTC time`

- **Generate Random Numbers**
  - `Give me a random number between 1 and 100`
  - `Pick a random number from 50 to 1000`

Ask me to perform any of these functions!
""")
            return
        
        # Empty query
        if not message:
            yield PartialResponse(text="Please enter a request. Type 'help' for instructions.")
            return
        
        # Send a MetaResponse with functions available
        yield MetaResponse(content={
            "functions": list(self.functions.values())
        })
        
        # Simulate a function call request based on the message
        # In a real implementation, the Poe platform would look at the meta response and
        # call our bot with the function to execute when appropriate
        function_call = self._determine_function_call(message)
        
        if function_call:
            function_name = function_call.get("name")
            function_params = function_call.get("parameters", {})
            
            # Show the function call (for demonstration)
            yield PartialResponse(text=f"```json\n{json.dumps({'function_call': function_call}, indent=2)}\n```\n\n")
            
            # Call the function
            result = self._call_function(function_name, function_params)
            
            # Format and return the result
            formatted_result = self._format_function_result(function_name, result)
            yield PartialResponse(text=formatted_result)
        else:
            yield PartialResponse(text="I'm not sure what function to call. Can you try being more specific?")
    
    def _determine_function_call(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Determine which function to call based on the message.
        This is a simple rule-based approach - in production, you might use a model.
        
        Args:
            message: The user's message
            
        Returns:
            Function call specification or None
        """
        message_lower = message.lower()
        
        # Calculate
        if re.search(r'calculate|compute|solve|what\'?s|how much is|evaluate', message_lower):
            # Extract expression
            expression = re.sub(r'(calculate|compute|solve|what\'?s|how much is|evaluate)', '', message_lower, flags=re.IGNORECASE).strip()
            expression = re.sub(r'[?]', '', expression)  # Remove question marks
            
            return {
                "name": "calculate",
                "parameters": {"expression": expression}
            }
        
        # Convert units
        elif re.search(r'convert|how many', message_lower):
            # Try to extract value and units
            match = re.search(r'(\d+(?:\.\d+)?)\s+([a-z]+)\s+(?:to|in)\s+([a-z]+)', message_lower)
            if match:
                value_str, from_unit, to_unit = match.groups()
                return {
                    "name": "convert_units",
                    "parameters": {
                        "value": float(value_str),
                        "from_unit": from_unit,
                        "to_unit": to_unit
                    }
                }
        
        # Get current time
        elif re.search(r'time|clock|hour|date', message_lower):
            timezone = "local"
            if re.search(r'utc|gmt|universal|greenwich', message_lower):
                timezone = "utc"
                
            return {
                "name": "get_current_time",
                "parameters": {"timezone": timezone}
            }
        
        # Generate random number
        elif re.search(r'random|pick|choose|generate', message_lower):
            # Try to extract min and max values
            match = re.search(r'(?:between|from)\s+(\d+)\s+(?:and|to)\s+(\d+)', message_lower)
            if match:
                min_str, max_str = match.groups()
                return {
                    "name": "generate_random_number",
                    "parameters": {
                        "min": int(min_str),
                        "max": int(max_str)
                    }
                }
            else:
                # Default range
                return {
                    "name": "generate_random_number",
                    "parameters": {
                        "min": 1,
                        "max": 100
                    }
                }
        
        return None
    
    def _format_function_result(self, function_name: str, result: Dict[str, Any]) -> str:
        """
        Format the function result into a readable response.
        
        Args:
            function_name: Name of the function called
            result: Result of the function call
            
        Returns:
            Formatted result message
        """
        if "error" in result:
            return f"Error: {result['error']}"
        
        if function_name == "calculate":
            return f"Result: {result['result']}"
        
        elif function_name == "convert_units":
            return f"{result['from_value']} {result['from_unit']} = {result['to_value']:.4f} {result['to_unit']}"
        
        elif function_name == "get_current_time":
            return f"Current time ({result['timezone']}): {result['time']}"
        
        elif function_name == "generate_random_number":
            return f"Random number between {result['min']} and {result['max']}: {result['random_number']}"
        
        return f"Function result: {json.dumps(result)}"