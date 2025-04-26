"""
File Analyzer Bot - A bot that analyzes uploaded files.

This bot demonstrates how to handle file uploads in a Poe bot,
including text extraction and analysis.
"""

import json
import logging
import os
import re
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi_poe.types import Attachment, PartialResponse, QueryRequest

from utils.base_bot import BaseBot, BotError, BotErrorNoRetry

logger = logging.getLogger(__name__)


class FileAnalyzerBot(BaseBot):
    """
    A bot that analyzes uploaded files.

    Supports:
    - Text files (.txt, .csv, .md, etc.)
    - Code files (.py, .js, .java, etc.)
    - Simple stats and analysis
    """

    bot_name = "FileAnalyzerBot"
    bot_description = (
        "A bot that analyzes uploaded files. Upload a file and I'll provide information about it."
    )
    version = "1.0.0"

    # Supported file types
    TEXT_EXTENSIONS = [".txt", ".md", ".csv", ".json", ".yaml", ".yml"]
    CODE_EXTENSIONS = [
        ".py",
        ".js",
        ".html",
        ".css",
        ".java",
        ".c",
        ".cpp",
        ".go",
        ".rs",
        ".php",
        ".rb",
    ]

    def __init__(self, **kwargs):
        """Initialize the FileAnalyzerBot."""
        super().__init__(**kwargs)

    def _extract_file_content(self, attachment: Attachment) -> str:
        """
        Extract content from a file attachment.

        Args:
            attachment: The file attachment

        Returns:
            The file content as text
        """
        try:
            # Get file extension
            file_name = attachment.name
            _, ext = os.path.splitext(file_name.lower())

            # Check if file type is supported
            supported_extensions = self.TEXT_EXTENSIONS + self.CODE_EXTENSIONS
            if ext not in supported_extensions:
                supported_list = ", ".join(supported_extensions)
                raise BotErrorNoRetry(
                    f"Unsupported file type: {ext}. Supported types: {supported_list}"
                )

            # Extract and return content
            # Access content via __dict__ to satisfy type checker
            # content attribute is added by the Poe platform but not in type definition
            if hasattr(attachment, "content") and attachment.__dict__.get("content"):
                content = attachment.__dict__["content"].decode("utf-8")
                return content
            else:
                raise BotErrorNoRetry("No content available in attachment")

        except UnicodeDecodeError:
            raise BotErrorNoRetry(
                "Unable to decode file. Make sure it's a text file with UTF-8 encoding."
            )
        except Exception as e:
            logger.error(f"Error extracting file content: {str(e)}")
            raise BotError(f"Error processing file: {str(e)}")

    def _analyze_file(self, content: str, file_name: str) -> Dict[str, Any]:
        """
        Analyze file content and return statistics.

        Args:
            content: The file content
            file_name: The file name

        Returns:
            Dictionary with file analysis
        """
        _, ext = os.path.splitext(file_name.lower())

        # Basic stats
        lines = content.split("\n")
        line_count = len(lines)
        word_count = len(re.findall(r"\b\w+\b", content))
        char_count = len(content)

        # Initialize analysis
        analysis = {
            "file_name": file_name,
            "file_extension": ext,
            "file_type": "text" if ext in self.TEXT_EXTENSIONS else "code",
            "line_count": line_count,
            "word_count": word_count,
            "character_count": char_count,
            "has_content": char_count > 0,
        }

        # Add file-specific analysis
        if ext == ".csv":
            analysis["csv_analysis"] = self._analyze_csv(content)
        elif ext == ".json":
            analysis["json_analysis"] = self._analyze_json(content)
        elif ext in self.CODE_EXTENSIONS:
            analysis["code_analysis"] = self._analyze_code(content, ext)

        return analysis

    def _analyze_csv(self, content: str) -> Dict[str, Any]:
        """Analyze CSV file content."""
        lines = content.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        if not non_empty_lines:
            return {"columns": 0, "rows": 0}

        header = non_empty_lines[0]
        columns = len(header.split(","))

        return {"columns": columns, "rows": len(non_empty_lines) - 1}  # Excluding header

    def _analyze_json(self, content: str) -> Dict[str, Any]:
        """Analyze JSON file content."""
        try:
            data = json.loads(content)

            if isinstance(data, list):
                return {"valid": True, "type": "array", "length": len(data)}
            elif isinstance(data, dict):
                return {
                    "valid": True,
                    "type": "object",
                    "keys": list(data.keys())[:10],  # First 10 keys
                    "key_count": len(data),
                }
            else:
                return {"valid": True, "type": type(data).__name__}

        except json.JSONDecodeError:
            return {"valid": False, "error": "Invalid JSON"}

    def _analyze_code(self, content: str, extension: str) -> Dict[str, Any]:
        """Analyze code file content."""
        language_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".html": "HTML",
            ".css": "CSS",
            ".java": "Java",
            ".c": "C",
            ".cpp": "C++",
            ".go": "Go",
            ".rs": "Rust",
            ".php": "PHP",
            ".rb": "Ruby",
        }

        language = language_map.get(extension, "Unknown")

        # Count comments
        comment_patterns = {
            "Python": r'(#[^\n]*|""".*?"""|\'\'\'.+?\'\'\')',
            "JavaScript": r"(//[^\n]*|/\*.*?\*/)",
            "HTML": r"<!--.*?-->",
            "CSS": r"/\*.*?\*/",
            "Java": r"(//[^\n]*|/\*.*?\*/)",
            "C": r"(//[^\n]*|/\*.*?\*/)",
            "C++": r"(//[^\n]*|/\*.*?\*/)",
            "Go": r"(//[^\n]*|/\*.*?\*/)",
            "Rust": r"(//[^\n]*|/\*.*?\*/)",
            "PHP": r"(//[^\n]*|/\*.*?\*/|#[^\n]*)",
            "Ruby": r"(#[^\n]*|=begin.*?=end)",
        }

        comment_count = 0
        if language in comment_patterns:
            # Simple comment counting - not perfect but good enough for demo
            comment_matches = re.findall(comment_patterns[language], content, re.DOTALL)
            comment_count = len(comment_matches)

        # Count functions/methods
        function_patterns = {
            "Python": r"def\s+\w+\s*\(",
            "JavaScript": r"function\s+\w+\s*\(|const\s+\w+\s*=\s*\([^)]*\)\s*=>",
            "Java": r"(public|private|protected)?\s+(static\s+)?\w+\s+\w+\s*\([^)]*\)\s*\{",
            "C": r"\w+\s+\w+\s*\([^)]*\)\s*\{",
            "C++": r"\w+\s+\w+\s*\([^)]*\)\s*\{",
            "Go": r"func\s+\w+\s*\(",
            "Rust": r"fn\s+\w+\s*\(",
            "PHP": r"function\s+\w+\s*\(",
            "Ruby": r"def\s+\w+\s*",
        }

        function_count = 0
        if language in function_patterns:
            function_matches = re.findall(function_patterns[language], content)
            function_count = len(function_matches)

        # Count imports/includes
        import_patterns = {
            "Python": r"import\s+\w+|from\s+\w+\s+import",
            "JavaScript": r"import\s+|require\(",
            "Java": r"import\s+[\w.]+;",
            "C": r'#include\s+[<"][^>"]+[>"]',
            "C++": r'#include\s+[<"][^>"]+[>"]',
            "Go": r'import\s+\([^)]+\)|import\s+"[^"]+"',
            "Rust": r"use\s+\w+",
            "PHP": r"require|include|require_once|include_once",
            "Ruby": r"require|include",
        }

        import_count = 0
        if language in import_patterns:
            import_matches = re.findall(import_patterns[language], content)
            import_count = len(import_matches)

        return {
            "language": language,
            "comment_count": comment_count,
            "function_count": function_count,
            "import_count": import_count,
        }

    def _format_analysis(self, analysis: Dict[str, Any], content: Optional[str] = None) -> str:
        """
        Format file analysis into a readable response.

        Args:
            analysis: The file analysis
            content: Optional file content for preview

        Returns:
            Formatted analysis text
        """
        file_name = analysis["file_name"]
        file_type = analysis["file_type"].capitalize()

        # Build the response
        response = f"## 📄 File Analysis: {file_name}\n\n"

        # Basic stats
        response += "### General Statistics\n"
        response += f"- **File Type:** {file_type} ({analysis['file_extension']})\n"
        response += f"- **Line Count:** {analysis['line_count']}\n"
        response += f"- **Word Count:** {analysis['word_count']}\n"
        response += f"- **Character Count:** {analysis['character_count']}\n\n"

        # Specific analysis based on file type
        if "csv_analysis" in analysis:
            csv_data = analysis["csv_analysis"]
            response += "### CSV Analysis\n"
            response += f"- **Columns:** {csv_data['columns']}\n"
            response += f"- **Rows:** {csv_data['rows']}\n\n"

        elif "json_analysis" in analysis:
            json_data = analysis["json_analysis"]
            response += "### JSON Analysis\n"
            response += f"- **Valid JSON:** {json_data['valid']}\n"

            if json_data.get("valid", False):
                response += f"- **Type:** {json_data['type']}\n"

                if json_data["type"] == "array":
                    response += f"- **Array Length:** {json_data['length']}\n"
                elif json_data["type"] == "object":
                    response += f"- **Number of Keys:** {json_data['key_count']}\n"

                    if "keys" in json_data and json_data["keys"]:
                        response += "- **Top Keys:**\n"
                        for key in json_data["keys"][:5]:  # Show first 5 keys
                            response += f"  - `{key}`\n"
            else:
                response += f"- **Error:** {json_data.get('error', 'Unknown error')}\n"

            response += "\n"

        elif "code_analysis" in analysis:
            code_data = analysis["code_analysis"]
            response += "### Code Analysis\n"
            response += f"- **Language:** {code_data['language']}\n"
            response += f"- **Function/Method Count:** {code_data['function_count']}\n"
            response += f"- **Import/Include Count:** {code_data['import_count']}\n"
            response += f"- **Comment Count:** {code_data['comment_count']}\n\n"

        # Add content preview
        if analysis["has_content"]:
            response += "### Content Preview\n"
            response += "```\n"
            # Get first 10 lines or 500 characters, whichever is shorter
            preview = "Content preview not available"
            if content:
                preview = "\n".join(content.split("\n")[:10])[:500]
            response += f"{preview}"
            response += "...\n```\n"

        return response

    async def _process_message(
        self, message: str, query: QueryRequest
    ) -> AsyncGenerator[PartialResponse, None]:
        """Process the user's message and handle file uploads."""
        message = message.strip()

        # Since we're working with Pydantic models, we need to access attachments correctly
        attachments = []
        if isinstance(query.query, list) and query.query:
            last_message = query.query[-1]
            if hasattr(last_message, "attachments"):
                attachments = last_message.attachments

        # Help command
        if message.lower() in ["help", "?", "/help"]:
            yield PartialResponse(
                text="""
## 📄 File Analyzer Bot

Upload a file, and I'll analyze it for you! I can provide statistics and insights for:

### Supported File Types

**Text Files:**
- .txt (Plain text)
- .md (Markdown)
- .csv (Comma-separated values)
- .json (JSON data)
- .yaml/.yml (YAML data)

**Code Files:**
- .py (Python)
- .js (JavaScript)
- .html (HTML)
- .css (CSS)
- .java (Java)
- .c (C)
- .cpp (C++)
- .go (Go)
- .rs (Rust)
- .php (PHP)
- .rb (Ruby)

Simply upload a file to get started!
"""
            )
            return

        # Check for file attachments
        if not attachments:
            yield PartialResponse(
                text="Please upload a file for me to analyze. Type 'help' for instructions."
            )
            return

        # Process the first attachment
        attachment = attachments[0]
        yield PartialResponse(text=f"Analyzing {attachment.name}...\n\n")

        try:
            # Extract content from file
            content = self._extract_file_content(attachment)

            # Analyze file
            analysis = self._analyze_file(content, attachment.name)

            # Format and return analysis
            formatted_analysis = self._format_analysis(analysis, content)
            yield PartialResponse(text=formatted_analysis)

        except BotErrorNoRetry as e:
            yield PartialResponse(text=f"Error: {str(e)}")
        except Exception as e:
            yield PartialResponse(text=f"Analysis error: {str(e)}")
            return
