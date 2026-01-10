"""
Unit tests for M3U8 parsing functions in m3u8-get.
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

parse_m3u_attributes = m3u8_get.parse_m3u_attributes
_parse_ext_x_media = m3u8_get._parse_ext_x_media
Track = m3u8_get.Track


class TestParseM3UAttributes:
    """Tests for parse_m3u_attributes function."""

    def test_parse_simple_attributes(self):
        """Test parsing simple key=value attributes."""
        line = '#EXT-X-STREAM-INF:BANDWIDTH=1000000,RESOLUTION=1920x1080'
        result = parse_m3u_attributes(line)
        assert result == {
            "BANDWIDTH": "1000000",
            "RESOLUTION": "1920x1080",
        }

    def test_parse_quoted_attributes(self):
        """Test parsing quoted attributes with commas."""
        line = '#EXT-X-STREAM-INF:CODECS="avc1.42e01e,mp4a.40.2",BANDWIDTH=1000000'
        result = parse_m3u_attributes(line)
        assert result == {
            "CODECS": "avc1.42e01e,mp4a.40.2",
            "BANDWIDTH": "1000000",
        }

    def test_parse_mixed_attributes(self):
        """Test parsing mix of quoted and unquoted attributes."""
        line = '#EXT-X-STREAM-INF:BANDWIDTH=1000000,RESOLUTION=1920x1080,CODECS="video,audio",AUDIO="audio"'
        result = parse_m3u_attributes(line)
        assert result == {
            "BANDWIDTH": "1000000",
            "RESOLUTION": "1920x1080",
            "CODECS": "video,audio",
            "AUDIO": "audio",
        }

    def test_parse_attributes_with_escape_sequences(self):
        """Test parsing attributes with escaped characters."""
        line = r'#EXT-X-STREAM-INF:NAME="Test\"Quote",BANDWIDTH=1000000'
        result = parse_m3u_attributes(line)
        # Note: The current implementation doesn't handle escaped quotes inside quoted strings
        # The regex captures 'Test\\' as the value
        assert result["NAME"] == 'Test\\'
        assert result["BANDWIDTH"] == "1000000"

    def test_parse_empty_attributes(self):
        """Test parsing a line with no attributes."""
        line = "#EXT-X-STREAM-INF:"
        result = parse_m3u_attributes(line)
        assert result == {}

    def test_parse_single_attribute(self):
        """Test parsing a single attribute."""
        line = "#EXT-X-STREAM-INF:BANDWIDTH=1000000"
        result = parse_m3u_attributes(line)
        assert result == {"BANDWIDTH": "1000000"}

    def test_parse_ext_x_media_attributes(self):
        """Test parsing EXT-X-MEDIA tag attributes."""
        line = '#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="English",DEFAULT=YES,LANGUAGE="en",URI="audio/en.m3u8"'
        result = parse_m3u_attributes(line)
        assert result == {
            "TYPE": "AUDIO",
            "GROUP-ID": "audio",
            "NAME": "English",
            "DEFAULT": "YES",
            "LANGUAGE": "en",
            "URI": "audio/en.m3u8",
        }

    def test_parse_attributes_with_spaces_in_quotes(self):
        """Test parsing attributes with spaces inside quotes."""
        line = '#EXT-X-MEDIA:NAME="Audio Track 1",TYPE=AUDIO,GROUP-ID="audio"'
        result = parse_m3u_attributes(line)
        assert result["NAME"] == "Audio Track 1"
        assert result["TYPE"] == "AUDIO"


class TestParseExtXMedia:
    """Tests for _parse_ext_x_media function."""

    def test_parse_audio_tracks(self, sample_m3u8_content):
        """Test parsing audio tracks from M3U8 content."""
        lines = sample_m3u8_content.split("\n")
        result = _parse_ext_x_media(lines, "http://example.com/")

        assert "audio" in result
        assert len(result["audio"]) == 2

        # Check first audio track
        audio_en = result["audio"][0]
        assert audio_en.type == "audio"
        assert audio_en.name == "English"
        assert audio_en.language == "en"
        assert audio_en.group_id == "audio"
        assert audio_en.url == "audio/en.m3u8"
        assert audio_en.is_default is True

        # Check second audio track
        audio_fr = result["audio"][1]
        assert audio_fr.type == "audio"
        assert audio_fr.name == "French"
        assert audio_fr.language == "fr"
        assert audio_fr.is_default is False

    def test_parse_subtitle_tracks(self, sample_m3u8_content):
        """Test parsing subtitle tracks from M3U8 content."""
        lines = sample_m3u8_content.split("\n")
        result = _parse_ext_x_media(lines, "http://example.com/")

        assert "subs" in result
        assert len(result["subs"]) == 1

        # Check subtitle track
        sub = result["subs"][0]
        assert sub.type == "subtitle"
        assert sub.name == "English"
        assert sub.language == "en"
        assert sub.group_id == "subs"
        assert sub.url == "subs/en.m3u8"

    def test_parse_empty_content(self):
        """Test parsing empty M3U8 content."""
        lines = []
        result = _parse_ext_x_media(lines, "http://example.com/")
        assert result == {}

    def test_parse_no_media_tags(self):
        """Test parsing content without EXT-X-MEDIA tags."""
        lines = ["#EXTM3U", "#EXT-X-VERSION:3", "segment1.ts"]
        result = _parse_ext_x_media(lines, "http://example.com/")
        assert result == {}

    def test_parse_with_channels(self):
        """Test parsing audio track with channel information."""
        content = """#EXTM3U
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="Stereo",CHANNELS="2",URI="audio/stereo.m3u8"
"""
        lines = content.split("\n")
        result = _parse_ext_x_media(lines, "http://example.com/")

        assert "audio" in result
        assert result["audio"][0].channels == "2"

    def test_parse_with_codecs(self):
        """Test parsing media track with codec information."""
        content = """#EXTM3U
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="AAC",CODECS="mp4a.40.2",URI="audio/aac.m3u8"
"""
        lines = content.split("\n")
        result = _parse_ext_x_media(lines, "http://example.com/")

        assert "audio" in result
        assert result["audio"][0].codec == "mp4a.40.2"

    def test_parse_video_type_ignored(self):
        """Test that VIDEO type in EXT-X-MEDIA is ignored (not standard)."""
        content = """#EXTM3U
#EXT-X-MEDIA:TYPE=VIDEO,GROUP-ID="video",NAME="Video",URI="video.m3u8"
"""
        lines = content.split("\n")
        result = _parse_ext_x_media(lines, "http://example.com/")

        # VIDEO type should be ignored (only audio and subtitles are supported)
        assert result == {}

    def test_parse_multiple_groups(self):
        """Test parsing multiple media groups."""
        content = """#EXTM3U
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio1",NAME="English",URI="audio/en.m3u8"
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio1",NAME="French",URI="audio/fr.m3u8"
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio2",NAME="Commentary",URI="audio/commentary.m3u8"
#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="English",URI="subs/en.m3u8"
"""
        lines = content.split("\n")
        result = _parse_ext_x_media(lines, "http://example.com/")

        assert "audio1" in result
        assert len(result["audio1"]) == 2
        assert "audio2" in result
        assert len(result["audio2"]) == 1
        assert "subs" in result
        assert len(result["subs"]) == 1

    def test_parse_without_uri(self):
        """Test parsing media track without URI (edge case)."""
        content = """#EXTM3U
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="English"
"""
        lines = content.split("\n")
        result = _parse_ext_x_media(lines, "http://example.com/")

        assert "audio" in result
        assert result["audio"][0].url == ""

    def test_parse_missing_group_id(self):
        """Test parsing media track without GROUP-ID."""
        content = """#EXTM3U
#EXT-X-MEDIA:TYPE=AUDIO,NAME="English",URI="audio/en.m3u8"
"""
        lines = content.split("\n")
        result = _parse_ext_x_media(lines, "http://example.com/")

        # Should not be added to result if no GROUP-ID
        assert result == {}
