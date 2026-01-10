"""
Pytest configuration and fixtures for m3u8-get tests.
"""

import sys
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the test helpers
import importlib.util
spec = importlib.util.spec_from_file_location("test_helpers", Path(__file__).parent / "test_helpers.py")
if spec is None:
    raise ImportError("Failed to load test_helpers module")
test_helpers = importlib.util.module_from_spec(spec)
if test_helpers is None:
    raise ImportError("Failed to create test_helpers module")
if spec.loader is None:
    raise ImportError("Failed to get loader for test_helpers")
spec.loader.exec_module(test_helpers)

Track = test_helpers.Track
MasterPlaylist = test_helpers.MasterPlaylist


@pytest.fixture
def sample_track():
    """Create a sample track for testing."""
    return Track(
        type="video",
        name="Test Video",
        language="en",
        group_id="video1",
        url="http://example.com/video.m3u8",
        bandwidth=1000000,
        resolution="1920x1080",
        is_default=True,
    )


@pytest.fixture
def sample_playlist():
    """Create a sample master playlist for testing."""
    return MasterPlaylist(
        tracks=[
            Track(
                type="video",
                name="Stream 1",
                bandwidth=1000000,
                resolution="1920x1080",
                url="http://example.com/video1.m3u8",
                index=0,
            )
        ],
        base_url="http://example.com/",
    )


@pytest.fixture
def sample_m3u8_content():
    """Sample M3U8 playlist content for testing."""
    return """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="English",DEFAULT=YES,LANGUAGE="en",URI="audio/en.m3u8"
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="French",LANGUAGE="fr",URI="audio/fr.m3u8"
#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="English",LANGUAGE="en",URI="subs/en.m3u8"
#EXT-X-STREAM-INF:BANDWIDTH=1000000,RESOLUTION=1920x1080,CODECS="avc1.42e01e,mp4a.40.2",AUDIO="audio",SUBTITLES="subs"
video1080p.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=500000,RESOLUTION=1280x720,CODECS="avc1.42e01e,mp4a.40.2",AUDIO="audio"
video720p.m3u8
"""


@pytest.fixture
def sample_m3u8_attributes():
    """Sample M3U8 attribute line for testing."""
    return '#EXT-X-STREAM-INF:BANDWIDTH=1000000,RESOLUTION=1920x1080,CODECS="avc1.42e01e,mp4a.40.2",AUDIO="audio"'
