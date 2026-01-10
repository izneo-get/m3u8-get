"""
Unit tests for utility functions in m3u8-get.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import importlib.util
spec = importlib.util.spec_from_file_location("m3u8_get", Path(__file__).parent.parent / "m3u8-get.py")
if spec is None:
    raise ImportError("Failed to load m3u8-get module spec")
m3u8_get = importlib.util.module_from_spec(spec)
if m3u8_get is None:
    raise ImportError("Failed to create m3u8-get module")
if spec.loader is None:
    raise ImportError("Failed to get loader for m3u8-get")
spec.loader.exec_module(m3u8_get)

is_valid_m3u = m3u8_get.is_valid_m3u
_get_base_url = m3u8_get._get_base_url
_normalize_url = m3u8_get._normalize_url


class TestIsValidM3U:
    """Tests for is_valid_m3u function."""

    def test_valid_m3u8_with_extm3u(self):
        """Test that valid M3U8 content is recognized."""
        content = "#EXTM3U\n#EXT-X-VERSION:3\n"
        assert is_valid_m3u(content) is True

    def test_valid_m3u8_lowercase(self):
        """Test that lowercase EXTM3U is NOT recognized (case-sensitive)."""
        content = "#extm3u\n#EXT-X-VERSION:3\n"
        # The function is case-sensitive, so lowercase should not be valid
        assert is_valid_m3u(content) is False

    def test_invalid_m3u8_without_header(self):
        """Test that content without EXTM3U is rejected."""
        content = "#EXT-X-VERSION:3\n"
        assert is_valid_m3u(content) is False

    def test_invalid_empty_content(self):
        """Test that empty content is rejected."""
        assert is_valid_m3u("") is False

    def test_invalid_random_content(self):
        """Test that random content is rejected."""
        assert is_valid_m3u("Hello World") is False


class TestGetBaseUrl:
    """Tests for _get_base_url function."""

    def test_get_base_url_simple(self):
        """Test extracting base URL from a simple URL."""
        url = "http://example.com/video.m3u8"
        result = _get_base_url(url)
        assert result == "http://example.com/"

    def test_get_base_url_with_path(self):
        """Test extracting base URL with nested path."""
        url = "http://example.com/path/to/video.m3u8"
        result = _get_base_url(url)
        assert result == "http://example.com/path/to/"

    def test_get_base_url_with_query_params(self):
        """Test extracting base URL with query parameters."""
        url = "http://example.com/video.m3u8?token=abc123"
        result = _get_base_url(url)
        assert result == "http://example.com/"

    def test_get_base_url_https(self):
        """Test extracting base URL from HTTPS URL."""
        url = "https://secure.example.com/stream.m3u8"
        result = _get_base_url(url)
        assert result == "https://secure.example.com/"

    def test_get_base_url_with_port(self):
        """Test extracting base URL with port number."""
        url = "http://example.com:8080/video.m3u8"
        result = _get_base_url(url)
        assert result == "http://example.com:8080/"

    def test_get_base_url_deep_nesting(self):
        """Test extracting base URL with deeply nested path."""
        url = "http://example.com/a/b/c/d/e/f/video.m3u8"
        result = _get_base_url(url)
        assert result == "http://example.com/a/b/c/d/e/f/"


class TestNormalizeUrl:
    """Tests for _normalize_url function."""

    def test_normalize_url_already_absolute_http(self):
        """Test that HTTP URLs are not modified."""
        url = "http://example.com/video.m3u8"
        base_url = "http://other.com/"
        result = _normalize_url(url, base_url)
        assert result == "http://example.com/video.m3u8"

    def test_normalize_url_already_absolute_https(self):
        """Test that HTTPS URLs are not modified."""
        url = "https://example.com/video.m3u8"
        base_url = "http://other.com/"
        result = _normalize_url(url, base_url)
        assert result == "https://example.com/video.m3u8"

    def test_normalize_url_relative_simple(self):
        """Test normalizing a simple relative URL."""
        url = "video.m3u8"
        base_url = "http://example.com/path/"
        result = _normalize_url(url, base_url)
        assert result == "http://example.com/path/video.m3u8"

    def test_normalize_url_relative_with_path(self):
        """Test normalizing a relative URL with path."""
        url = "../video.m3u8"
        base_url = "http://example.com/path/to/"
        result = _normalize_url(url, base_url)
        assert result == "http://example.com/path/video.m3u8"

    def test_normalize_url_relative_subdirectory(self):
        """Test normalizing a relative URL to subdirectory."""
        url = "streams/video.m3u8"
        base_url = "http://example.com/"
        result = _normalize_url(url, base_url)
        assert result == "http://example.com/streams/video.m3u8"

    def test_normalize_url_absolute_path(self):
        """Test normalizing an absolute path URL."""
        url = "/video.m3u8"
        base_url = "http://example.com/path/"
        result = _normalize_url(url, base_url)
        assert result == "http://example.com/video.m3u8"

    def test_normalize_url_empty_string(self):
        """Test normalizing an empty URL."""
        url = ""
        base_url = "http://example.com/"
        result = _normalize_url(url, base_url)
        assert result == "http://example.com/"

    def test_normalize_url_with_query_params(self):
        """Test normalizing URL with query parameters."""
        url = "video.m3u8?token=abc"
        base_url = "http://example.com/"
        result = _normalize_url(url, base_url)
        assert result == "http://example.com/video.m3u8?token=abc"

    def test_normalize_url_with_fragment(self):
        """Test normalizing URL with fragment."""
        url = "video.m3u8#section"
        base_url = "http://example.com/"
        result = _normalize_url(url, base_url)
        assert result == "http://example.com/video.m3u8#section"
