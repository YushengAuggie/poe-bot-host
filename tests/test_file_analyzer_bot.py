"""
Tests for the FileAnalyzerBot implementation.
"""

import pytest
from fastapi_poe.types import Attachment, QueryRequest

from bots.file_analyzer_bot import FileAnalyzerBot
from utils.base_bot import BotError


# Mock file attachment
class MockAttachment(Attachment):
    url: str = "mock://file"
    content_type: str = "application/octet-stream"
    content: bytes = b""

    def __init__(self, name: str, content: bytes):
        super().__init__(url="mock://file", content_type="application/octet-stream", name=name)
        # Store content as a non-model field
        object.__setattr__(self, "content", content)


@pytest.fixture
def file_analyzer_bot():
    """Create a FileAnalyzerBot instance for testing."""
    return FileAnalyzerBot()


@pytest.fixture
def txt_file_attachment():
    """Create a mock text file attachment."""
    content = b"This is a test file.\nIt has multiple lines.\nTotal of 3 lines."
    return MockAttachment("test.txt", content)


@pytest.fixture
def py_file_attachment():
    """Create a mock Python file attachment."""
    content = b"""#!/usr/bin/env python
\"\"\"
Test Python file.
\"\"\"

import os
import sys

def hello_world():
    \"\"\"Print hello world.\"\"\"
    print("Hello, world!")

if __name__ == "__main__":
    hello_world()
"""
    return MockAttachment("test.py", content)


@pytest.fixture
def csv_file_attachment():
    """Create a mock CSV file attachment."""
    content = b"""name,age,city
Alice,30,New York
Bob,25,Los Angeles
Charlie,35,Chicago
"""
    return MockAttachment("test.csv", content)


@pytest.fixture
def json_file_attachment():
    """Create a mock JSON file attachment."""
    content = b"""{
    "users": [
        {"name": "Alice", "age": 30, "city": "New York"},
        {"name": "Bob", "age": 25, "city": "Los Angeles"},
        {"name": "Charlie", "age": 35, "city": "Chicago"}
    ],
    "version": "1.0"
}"""
    return MockAttachment("test.json", content)


@pytest.fixture
def invalid_file_attachment():
    """Create a mock unsupported file attachment."""
    content = b"Binary content \x00\x01\x02"
    # Use .bin extension which is not in the supported extensions list
    return MockAttachment("test.bin", content)


@pytest.fixture
def sample_query_with_attachment(txt_file_attachment):
    """Create a sample query with a file attachment."""
    from fastapi_poe.types import ProtocolMessage

    # Create a protocol message with an attachment
    message = ProtocolMessage(
        role="user", content="Analyze this file", attachments=[txt_file_attachment]
    )

    # Create the QueryRequest with the protocol message
    return QueryRequest(
        version="1.0",
        type="query",
        query=[message],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )


@pytest.mark.asyncio
async def test_file_analyzer_initialization(file_analyzer_bot):
    """Test FileAnalyzerBot initialization."""
    # Check the class attributes rather than instance attributes
    assert FileAnalyzerBot.bot_name == "FileAnalyzerBot"
    assert "file" in FileAnalyzerBot.bot_description.lower()
    assert "analyze" in FileAnalyzerBot.bot_description.lower()


@pytest.mark.asyncio
async def test_help_command(file_analyzer_bot):
    """Test file analyzer help command."""
    from fastapi_poe.types import ProtocolMessage

    # Create a protocol message
    message = ProtocolMessage(role="user", content="help")

    query = QueryRequest(
        version="1.0",
        type="query",
        query=[message],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )

    responses = []
    async for response in file_analyzer_bot._process_message("help", query):
        responses.append(response)

    # Verify help content is returned
    assert len(responses) == 1
    assert "File Analyzer Bot" in responses[0].text
    assert "Supported File Types" in responses[0].text


@pytest.mark.asyncio
async def test_empty_query_no_attachment(file_analyzer_bot):
    """Test file analyzer with no file attachment."""
    from fastapi_poe.types import ProtocolMessage

    # Create a protocol message with no attachments
    message = ProtocolMessage(role="user", content="")

    query = QueryRequest(
        version="1.0",
        type="query",
        query=[message],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )

    responses = []
    async for response in file_analyzer_bot._process_message("", query):
        responses.append(response)

    # Verify prompt for file upload is returned
    assert len(responses) == 1
    assert "Please upload a file" in responses[0].text


@pytest.mark.asyncio
async def test_extract_text_file_content(file_analyzer_bot, txt_file_attachment):
    """Test extracting content from a text file."""
    content = file_analyzer_bot._extract_file_content(txt_file_attachment)

    assert content == "This is a test file.\nIt has multiple lines.\nTotal of 3 lines."


@pytest.mark.asyncio
async def test_extract_python_file_content(file_analyzer_bot, py_file_attachment):
    """Test extracting content from a Python file."""
    content = file_analyzer_bot._extract_file_content(py_file_attachment)

    assert "def hello_world():" in content
    assert 'print("Hello, world!")' in content


@pytest.mark.asyncio
async def test_extract_unsupported_file(file_analyzer_bot, invalid_file_attachment):
    """Test extracting content from an unsupported file type."""
    with pytest.raises(BotError) as excinfo:
        file_analyzer_bot._extract_file_content(invalid_file_attachment)

    assert "Unsupported file type" in str(excinfo.value)


@pytest.mark.asyncio
async def test_analyze_text_file(file_analyzer_bot, txt_file_attachment):
    """Test analyzing a text file."""
    # Extract content first
    content = file_analyzer_bot._extract_file_content(txt_file_attachment)

    # Analyze the file
    analysis = file_analyzer_bot._analyze_file(content, txt_file_attachment.name)

    # Check basic stats
    assert analysis["file_name"] == "test.txt"
    assert analysis["file_extension"] == ".txt"
    assert analysis["file_type"] == "text"
    assert analysis["line_count"] == 3
    assert (
        analysis["word_count"] == 13
    )  # This is a test file. It has multiple lines. Total of 3 lines.
    # The character count can vary slightly depending on line ending conversions
    assert 60 <= analysis["character_count"] <= 65  # Allow for some flexibility in newline handling
    assert analysis["has_content"] is True


@pytest.mark.asyncio
async def test_analyze_python_file(file_analyzer_bot, py_file_attachment):
    """Test analyzing a Python file."""
    # Extract content first
    content = file_analyzer_bot._extract_file_content(py_file_attachment)

    # Analyze the file
    analysis = file_analyzer_bot._analyze_file(content, py_file_attachment.name)

    # Check basic stats
    assert analysis["file_name"] == "test.py"
    assert analysis["file_extension"] == ".py"
    assert analysis["file_type"] == "code"
    assert analysis["line_count"] > 5
    assert analysis["has_content"] is True

    # Check code analysis
    assert "code_analysis" in analysis
    assert analysis["code_analysis"]["language"] == "Python"
    assert analysis["code_analysis"]["function_count"] >= 1  # Should find hello_world()
    assert analysis["code_analysis"]["import_count"] >= 2  # Should find import os and import sys
    assert (
        analysis["code_analysis"]["comment_count"] >= 1
    )  # Should find at least the module docstring


@pytest.mark.asyncio
async def test_analyze_csv_file(file_analyzer_bot, csv_file_attachment):
    """Test analyzing a CSV file."""
    # Extract content first
    content = file_analyzer_bot._extract_file_content(csv_file_attachment)

    # Analyze the file
    analysis = file_analyzer_bot._analyze_file(content, csv_file_attachment.name)

    # Check basic stats
    assert analysis["file_name"] == "test.csv"
    assert analysis["file_extension"] == ".csv"
    assert analysis["file_type"] == "text"
    assert analysis["has_content"] is True

    # Check CSV analysis
    assert "csv_analysis" in analysis
    assert analysis["csv_analysis"]["columns"] == 3  # name,age,city
    assert analysis["csv_analysis"]["rows"] == 3  # Three data rows


@pytest.mark.asyncio
async def test_analyze_json_file(file_analyzer_bot, json_file_attachment):
    """Test analyzing a JSON file."""
    # Extract content first
    content = file_analyzer_bot._extract_file_content(json_file_attachment)

    # Analyze the file
    analysis = file_analyzer_bot._analyze_file(content, json_file_attachment.name)

    # Check basic stats
    assert analysis["file_name"] == "test.json"
    assert analysis["file_extension"] == ".json"
    assert analysis["file_type"] == "text"
    assert analysis["has_content"] is True

    # Check JSON analysis
    assert "json_analysis" in analysis
    assert analysis["json_analysis"]["valid"] is True
    assert analysis["json_analysis"]["type"] == "object"
    assert analysis["json_analysis"]["key_count"] == 2  # users and version
    assert "users" in analysis["json_analysis"]["keys"]
    assert "version" in analysis["json_analysis"]["keys"]


@pytest.mark.asyncio
async def test_format_analysis(file_analyzer_bot):
    """Test formatting analysis results."""
    # Create a simple analysis result
    analysis = {
        "file_name": "test.txt",
        "file_extension": ".txt",
        "file_type": "text",
        "line_count": 3,
        "word_count": 10,
        "character_count": 50,
        "has_content": True,
    }

    # Sample content to pass to the method
    sample_content = "This is sample content.\nLine 2\nLine 3"

    formatted = file_analyzer_bot._format_analysis(analysis, sample_content)

    # Check formatting
    assert "## 📄 File Analysis: test.txt" in formatted
    assert "**File Type:** Text (.txt)" in formatted
    assert "**Line Count:** 3" in formatted
    assert "**Word Count:** 10" in formatted
    assert "**Character Count:** 50" in formatted
    assert "This is sample content." in formatted


@pytest.mark.asyncio
async def test_process_file(file_analyzer_bot, sample_query_with_attachment):
    """Test processing a file attachment."""
    message = "Analyze this file"

    responses = []
    async for response in file_analyzer_bot._process_message(message, sample_query_with_attachment):
        responses.append(response)

    # Should have two responses: "Analyzing..." and the analysis
    assert len(responses) == 2
    assert "Analyzing" in responses[0].text
    assert "File Analysis:" in responses[1].text
