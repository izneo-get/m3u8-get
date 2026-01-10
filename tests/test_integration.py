"""
Integration tests for complete parsing functions in m3u8-get.
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

_parse_ext_x_stream_inf = m3u8_get._parse_ext_x_stream_inf
_handle_standalone_media_tracks = m3u8_get._handle_standalone_media_tracks
Track = m3u8_get.Track
MasterPlaylist = m3u8_get.MasterPlaylist
_parse_ext_x_media = m3u8_get._parse_ext_x_media


class TestParseExtXStreamInf:
    """Tests for _parse_ext_x_stream_inf function."""

    def test_parse_single_video_stream(self):
        """Test parsing a single video stream."""
        content = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=1000000,RESOLUTION=1920x1080
video1080p.m3u8
"""
        lines = content.split("\n")
        media_tracks = {}
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")

        _parse_ext_x_stream_inf(lines, media_tracks, playlist)

        assert len(playlist.tracks) == 1
        assert playlist.tracks[0].type == "video"
        assert playlist.tracks[0].bandwidth == 1000000
        assert playlist.tracks[0].resolution == "1920x1080"
        assert playlist.tracks[0].url == "http://example.com/video1080p.m3u8"

    def test_parse_multiple_video_streams(self):
        """Test parsing multiple video streams."""
        content = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=1000000,RESOLUTION=1920x1080
video1080p.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=500000,RESOLUTION=1280x720
video720p.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=250000,RESOLUTION=640x360
video360p.m3u8
"""
        lines = content.split("\n")
        media_tracks = {}
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")

        _parse_ext_x_stream_inf(lines, media_tracks, playlist)

        assert len(playlist.tracks) == 3
        assert playlist.tracks[0].resolution == "1920x1080"
        assert playlist.tracks[1].resolution == "1280x720"
        assert playlist.tracks[2].resolution == "640x360"

    def test_parse_with_absolute_url(self):
        """Test parsing stream with absolute URL."""
        content = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=1000000
http://cdn.example.com/video.m3u8
"""
        lines = content.split("\n")
        media_tracks = {}
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")

        _parse_ext_x_stream_inf(lines, media_tracks, playlist)

        assert len(playlist.tracks) == 1
        assert playlist.tracks[0].url == "http://cdn.example.com/video.m3u8"

    def test_parse_with_audio_associations(self, sample_m3u8_content):
        """Test parsing video streams with associated audio tracks."""
        lines = sample_m3u8_content.split("\n")

        # First, parse the media tracks
        media_tracks = _parse_ext_x_media(lines, "http://example.com/")

        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")

        _parse_ext_x_stream_inf(lines, media_tracks, playlist)

        # Should have 2 video tracks
        video_tracks = [t for t in playlist.tracks if t.type == "video"]
        assert len(video_tracks) == 2

        # Should have audio tracks associated
        audio_tracks = [t for t in playlist.tracks if t.type == "audio"]
        assert len(audio_tracks) == 2  # English and French

        # Check that audio URLs were normalized
        for audio in audio_tracks:
            assert audio.url.startswith("http://example.com/")

    def test_parse_with_subtitle_associations(self, sample_m3u8_content):
        """Test parsing video streams with associated subtitle tracks."""
        lines = sample_m3u8_content.split("\n")

        # First, parse the media tracks
        media_tracks = _parse_ext_x_media(lines, "http://example.com/")

        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")

        _parse_ext_x_stream_inf(lines, media_tracks, playlist)

        # Should have subtitle tracks
        sub_tracks = [t for t in playlist.tracks if t.type == "subtitle"]
        assert len(sub_tracks) == 1
        assert sub_tracks[0].name == "English"

    def test_parse_stream_index_increments(self):
        """Test that stream index increments correctly."""
        content = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=1000000
video1.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=500000
video2.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=250000
video3.m3u8
"""
        lines = content.split("\n")
        media_tracks = {}
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")

        _parse_ext_x_stream_inf(lines, media_tracks, playlist)

        assert playlist.tracks[0].index == 0
        assert playlist.tracks[1].index == 1
        assert playlist.tracks[2].index == 2

    def test_parse_with_codecs(self):
        """Test parsing stream with codec information."""
        content = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=1000000,CODECS="avc1.42e01e,mp4a.40.2"
video.m3u8
"""
        lines = content.split("\n")
        media_tracks = {}
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")

        _parse_ext_x_stream_inf(lines, media_tracks, playlist)

        assert playlist.tracks[0].codec == "avc1.42e01e,mp4a.40.2"

    def test_parse_skips_comment_lines(self):
        """Test that comment lines after STREAM-INF are skipped."""
        content = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=1000000
# This is a comment
#EXT-X-STREAM-INF:BANDWIDTH=500000
video720p.m3u8
"""
        lines = content.split("\n")
        media_tracks = {}
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")

        _parse_ext_x_stream_inf(lines, media_tracks, playlist)

        # Only the second stream should be parsed (first has no URL)
        assert len(playlist.tracks) == 1
        assert playlist.tracks[0].bandwidth == 500000

    def test_parse_empty_url(self):
        """Test that empty URLs are skipped."""
        content = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=1000000

#EXT-X-STREAM-INF:BANDWIDTH=500000
video.m3u8
"""
        lines = content.split("\n")
        media_tracks = {}
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")

        _parse_ext_x_stream_inf(lines, media_tracks, playlist)

        # Only the second stream should be parsed
        assert len(playlist.tracks) == 1

    def test_parse_with_relative_url_normalization(self):
        """Test that relative URLs are normalized."""
        content = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=1000000
../video.m3u8
"""
        lines = content.split("\n")
        media_tracks = {}
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/path/to/")

        _parse_ext_x_stream_inf(lines, media_tracks, playlist)

        assert playlist.tracks[0].url == "http://example.com/path/video.m3u8"

    def test_parse_no_video_streams(self):
        """Test parsing content with no video streams."""
        content = """#EXTM3U
#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="English",URI="audio.m3u8
#EXT-X-VERSION:3
"""
        lines = content.split("\n")
        media_tracks = {}
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")

        _parse_ext_x_stream_inf(lines, media_tracks, playlist)

        # No video tracks should be added
        assert len(playlist.tracks) == 0


class TestHandleStandaloneMediaTracks:
    """Tests for _handle_standalone_media_tracks function."""

    def test_handle_audio_tracks_no_video(self):
        """Test handling audio tracks when no video is present."""
        audio_tracks = [
            Track(type="audio", name="English", group_id="audio1", url="audio/en.m3u8"),
            Track(type="audio", name="French", group_id="audio1", url="audio/fr.m3u8"),
        ]
        media_tracks = {"audio1": audio_tracks}
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")

        _handle_standalone_media_tracks(media_tracks, playlist)

        assert len(playlist.tracks) == 2
        assert playlist.tracks[0].type == "audio"
        assert playlist.tracks[0].url == "http://example.com/audio/en.m3u8"
        assert playlist.tracks[1].url == "http://example.com/audio/fr.m3u8"

    def test_handle_subtitle_tracks_no_video(self):
        """Test handling subtitle tracks when no video is present."""
        sub_tracks = [
            Track(type="subtitle", name="English", group_id="subs1", url="subs/en.m3u8")
        ]
        media_tracks = {"subs1": sub_tracks}
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")

        _handle_standalone_media_tracks(media_tracks, playlist)

        assert len(playlist.tracks) == 1
        assert playlist.tracks[0].type == "subtitle"
        assert playlist.tracks[0].url == "http://example.com/subs/en.m3u8"

    def test_handle_with_absolute_urls(self):
        """Test that absolute URLs are preserved."""
        audio_tracks = [
            Track(
                type="audio",
                name="English",
                group_id="audio1",
                url="http://cdn.example.com/audio.m3u8",
            )
        ]
        media_tracks = {"audio1": audio_tracks}
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")

        _handle_standalone_media_tracks(media_tracks, playlist)

        assert playlist.tracks[0].url == "http://cdn.example.com/audio.m3u8"

    def test_handle_with_relative_urls(self):
        """Test that relative URLs are normalized."""
        audio_tracks = [
            Track(type="audio", name="English", group_id="audio1", url="../audio/en.m3u8")
        ]
        media_tracks = {"audio1": audio_tracks}
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/path/to/")

        _handle_standalone_media_tracks(media_tracks, playlist)

        assert playlist.tracks[0].url == "http://example.com/path/audio/en.m3u8"

    def test_handle_multiple_groups(self):
        """Test handling tracks from multiple groups."""
        audio_tracks = [
            Track(type="audio", name="English", group_id="audio1", url="audio/en.m3u8")
        ]
        sub_tracks = [
            Track(type="subtitle", name="English", group_id="subs1", url="subs/en.m3u8")
        ]
        media_tracks = {"audio1": audio_tracks, "subs1": sub_tracks}
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")

        _handle_standalone_media_tracks(media_tracks, playlist)

        assert len(playlist.tracks) == 2
        assert playlist.tracks[0].type == "audio"
        assert playlist.tracks[1].type == "subtitle"

    def test_handle_empty_media_tracks(self):
        """Test with empty media tracks dictionary."""
        media_tracks = {}
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")

        _handle_standalone_media_tracks(media_tracks, playlist)

        assert len(playlist.tracks) == 0

    def test_handle_no_duplicate_tracks(self):
        """Test that duplicate tracks are not added twice."""
        existing_track = Track(type="audio", name="English", url="http://example.com/audio/en.m3u8")
        playlist = MasterPlaylist(tracks=[existing_track], base_url="http://example.com/")

        # Create the exact same track (same URL and same group_id for equality)
        audio_tracks = [
            Track(type="audio", name="English", group_id="", url="http://example.com/audio/en.m3u8")
        ]
        media_tracks = {"audio1": audio_tracks}

        _handle_standalone_media_tracks(media_tracks, playlist)

        # Should still have only 1 track (no duplicate)
        assert len(playlist.tracks) == 1

    def test_handle_with_empty_url(self):
        """Test handling tracks with empty URLs."""
        audio_tracks = [
            Track(type="audio", name="English", group_id="audio1", url="")
        ]
        media_tracks = {"audio1": audio_tracks}
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")

        _handle_standalone_media_tracks(media_tracks, playlist)

        # Empty URL should still be added (edge case)
        assert len(playlist.tracks) == 1
        assert playlist.tracks[0].url == "http://example.com/"
