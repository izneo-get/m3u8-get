"""
Unit tests for headers loading in m3u8-get.
"""

import sys
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util
import pytest

# Import the module
spec = importlib.util.spec_from_file_location("m3u8_get", Path(__file__).parent.parent / "m3u8-get.py")
if spec is None:
    raise ImportError("Failed to load m3u8-get module spec")
m3u8_get = importlib.util.module_from_spec(spec)
if m3u8_get is None:
    raise ImportError("Failed to create m3u8-get module")
if spec.loader is None:
    raise ImportError("Failed to get loader for m3u8-get")
spec.loader.exec_module(m3u8_get)

load_headers = m3u8_get.load_headers
DEFAULT_HEADERS = m3u8_get.DEFAULT_HEADERS

class TestLoadHeaders:
    """Tests for load_headers function."""
    
    def test_load_headers_json(self):
        """Test loading headers from a JSON file."""
        headers_data = {"User-Agent": "TestAgent", "Authorization": "Bearer token"}
        json_content = json.dumps(headers_data)
        
        with patch("builtins.open", mock_open(read_data=json_content)):
            with patch("os.path.isfile", return_value=True):
                headers = load_headers("headers.json")
                assert headers == headers_data
                
    def test_load_headers_python_dict(self):
        """Test loading headers from a file with Python dict syntax."""
        dict_content = "{'User-Agent': 'TestAgent', 'Authorization': 'Bearer token'}"
        expected_headers = {"User-Agent": "TestAgent", "Authorization": "Bearer token"}
        
        with patch("builtins.open", mock_open(read_data=dict_content)):
            with patch("os.path.isfile", return_value=True):
                headers = load_headers("headers.txt")
                assert headers == expected_headers

    def test_load_headers_invalid_syntax(self):
        """Test loading headers from a file with invalid syntax."""
        invalid_content = "{'User-Agent': "  # Incomplete
        
        with patch("builtins.open", mock_open(read_data=invalid_content)):
            with patch("os.path.isfile", return_value=True):
                # Should return default headers (and print error, but we don't check print here)
                headers = load_headers("headers.txt")
                assert headers == DEFAULT_HEADERS

    def test_load_headers_not_a_dict(self):
        """Test loading a file that evaluates to something else (e.g. list)."""
        list_content = "['item1', 'item2']"
        
        with patch("builtins.open", mock_open(read_data=list_content)):
            with patch("os.path.isfile", return_value=True):
                headers = load_headers("headers.txt")
                assert headers == DEFAULT_HEADERS

    def test_load_headers_file_not_found(self):
        """Test when headers file does not exist."""
        with patch("os.path.isfile", return_value=False):
            headers = load_headers("nonexistent.json")
            assert headers == DEFAULT_HEADERS
