"""
Unit tests for track creation and management functions in m3u8-get.
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

_create_video_track = m3u8_get._create_video_track
_add_associated_media_tracks = m3u8_get._add_associated_media_tracks
Track = m3u8_get.Track
MasterPlaylist = m3u8_get.MasterPlaylist


class TestCreateVideoTrack:
    """Tests for _create_video_track function."""

    def test_create_basic_video_track(self):
        """Test creating a basic video track."""
        attrs = {"BANDWIDTH": "1000000", "RESOLUTION": "1920x1080"}
        url = "http://example.com/video.m3u8"
        index = 0

        track = _create_video_track(attrs, url, index)

        assert track.type == "video"
        assert track.name == "Stream 1"
        assert track.bandwidth == 1000000
        assert track.resolution == "1920x1080"
        assert track.url == url
        assert track.index == 0

    def test_create_video_track_with_codecs(self):
        """Test creating video track with codec information."""
        attrs = {
            "BANDWIDTH": "2000000",
            "RESOLUTION": "1920x1080",
            "CODECS": "avc1.42e01e,mp4a.40.2",
        }
        url = "http://example.com/video2.m3u8"
        index = 1

        track = _create_video_track(attrs, url, index)

        assert track.codec == "avc1.42e01e,mp4a.40.2"
        assert track.bandwidth == 2000000
        assert track.index == 1

    def test_create_video_track_missing_bandwidth(self):
        """Test creating video track with missing bandwidth."""
        attrs = {"RESOLUTION": "1280x720"}
        url = "http://example.com/video.m3u8"
        index = 0

        track = _create_video_track(attrs, url, index)

        assert track.bandwidth == 0

    def test_create_video_track_missing_resolution(self):
        """Test creating video track with missing resolution."""
        attrs = {"BANDWIDTH": "500000"}
        url = "http://example.com/video.m3u8"
        index = 0

        track = _create_video_track(attrs, url, index)

        assert track.resolution == ""

    def test_create_video_track_index_in_name(self):
        """Test that track name includes index correctly."""
        attrs = {"BANDWIDTH": "1000000"}
        url = "http://example.com/video.m3u8"

        track0 = _create_video_track(attrs, url, 0)
        assert track0.name == "Stream 1"

        track5 = _create_video_track(attrs, url, 5)
        assert track5.name == "Stream 6"

    def test_create_video_track_all_attributes(self):
        """Test creating video track with all attributes."""
        attrs = {
            "BANDWIDTH": "5000000",
            "RESOLUTION": "3840x2160",
            "CODECS": "hevc,mp4a.40.2",
        }
        url = "http://example.com/4k.m3u8"
        index = 2

        track = _create_video_track(attrs, url, index)

        assert track.type == "video"
        assert track.name == "Stream 3"
        assert track.bandwidth == 5000000
        assert track.resolution == "3840x2160"
        assert track.codec == "hevc,mp4a.40.2"
        assert track.url == url
        assert track.index == 2


class TestAddAssociatedMediaTracks:
    """Tests for _add_associated_media_tracks function."""

    def test_add_audio_tracks_to_playlist(self, sample_playlist):
        """Test adding audio tracks to playlist."""
        audio_tracks = [
            Track(
                type="audio",
                name="English",
                language="en",
                group_id="audio1",
                url="audio/en.m3u8",
            ),
            Track(
                type="audio",
                name="French",
                language="fr",
                group_id="audio1",
                url="audio/fr.m3u8",
            ),
        ]
        media_tracks = {"audio1": audio_tracks}

        _add_associated_media_tracks("audio1", media_tracks, sample_playlist)

        # Check that audio tracks were added
        assert len(sample_playlist.tracks) == 3  # 1 video + 2 audio
        assert sample_playlist.tracks[1].type == "audio"
        assert sample_playlist.tracks[1].name == "English"
        assert sample_playlist.tracks[2].type == "audio"
        assert sample_playlist.tracks[2].name == "French"

    def test_add_subtitle_tracks_to_playlist(self, sample_playlist):
        """Test adding subtitle tracks to playlist."""
        sub_tracks = [
            Track(
                type="subtitle",
                name="English",
                language="en",
                group_id="subs1",
                url="subs/en.m3u8",
            )
        ]
        media_tracks = {"subs1": sub_tracks}

        _add_associated_media_tracks("subs1", media_tracks, sample_playlist)

        # Check that subtitle track was added
        assert len(sample_playlist.tracks) == 2  # 1 video + 1 subtitle
        assert sample_playlist.tracks[1].type == "subtitle"
        assert sample_playlist.tracks[1].name == "English"

    def test_normalize_relative_url(self):
        """Test that relative URLs are normalized."""
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/path/")
        audio_tracks = [
            Track(type="audio", name="Audio", group_id="audio1", url="../audio/en.m3u8")
        ]
        media_tracks = {"audio1": audio_tracks}

        _add_associated_media_tracks("audio1", media_tracks, playlist)

        assert playlist.tracks[0].url == "http://example.com/audio/en.m3u8"

    def test_normalize_absolute_url_unchanged(self):
        """Test that absolute URLs are not modified."""
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")
        audio_tracks = [
            Track(type="audio", name="Audio", group_id="audio1", url="http://other.com/audio.m3u8")
        ]
        media_tracks = {"audio1": audio_tracks}

        _add_associated_media_tracks("audio1", media_tracks, playlist)

        assert playlist.tracks[0].url == "http://other.com/audio.m3u8"

    def test_empty_group_id(self, sample_playlist):
        """Test with empty group ID - should not add anything."""
        media_tracks = {"audio1": []}

        _add_associated_media_tracks("", media_tracks, sample_playlist)

        # No tracks should be added
        assert len(sample_playlist.tracks) == 1  # Still just the original video

    def test_nonexistent_group_id(self, sample_playlist):
        """Test with group ID that doesn't exist - should not add anything."""
        media_tracks = {"audio1": []}

        _add_associated_media_tracks("audio2", media_tracks, sample_playlist)

        # No tracks should be added
        assert len(sample_playlist.tracks) == 1  # Still just the original video

    def test_no_duplicate_tracks(self, sample_playlist):
        """Test that duplicate tracks are not added twice."""
        existing_track = Track(type="audio", name="English", url="http://example.com/en.m3u8")
        sample_playlist.tracks.append(existing_track)

        audio_tracks = [
            Track(type="audio", name="English", url="http://example.com/en.m3u8")
        ]
        media_tracks = {"audio1": audio_tracks}

        _add_associated_media_tracks("audio1", media_tracks, sample_playlist)

        # Should still have only 2 tracks (video + one audio)
        assert len(sample_playlist.tracks) == 2

    def test_add_multiple_tracks_from_group(self):
        """Test adding multiple tracks from the same group."""
        playlist = MasterPlaylist(tracks=[], base_url="http://example.com/")
        tracks = [
            Track(type="audio", name="English", group_id="audio1", url="audio/en.m3u8"),
            Track(type="audio", name="French", group_id="audio1", url="audio/fr.m3u8"),
            Track(type="audio", name="Spanish", group_id="audio1", url="audio/es.m3u8"),
        ]
        media_tracks = {"audio1": tracks}

        _add_associated_media_tracks("audio1", media_tracks, playlist)

        assert len(playlist.tracks) == 3
        assert playlist.tracks[0].name == "English"
        assert playlist.tracks[1].name == "French"
        assert playlist.tracks[2].name == "Spanish"

    def test_url_normalization_https(self):
        """Test URL normalization with HTTPS."""
        playlist = MasterPlaylist(tracks=[], base_url="https://example.com/video/")
        audio_tracks = [
            Track(type="audio", name="Audio", group_id="audio1", url="../audio/en.m3u8")
        ]
        media_tracks = {"audio1": audio_tracks}

        _add_associated_media_tracks("audio1", media_tracks, playlist)

        assert playlist.tracks[0].url == "https://example.com/audio/en.m3u8"
