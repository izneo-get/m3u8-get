"""
Helper functions to load m3u8-get module for testing.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import all functions and classes from m3u8-get module
import importlib.util

spec = importlib.util.spec_from_file_location("m3u8_get", Path(__file__).parent.parent / "m3u8-get.py")
if spec is None:
    raise ImportError("Failed to load m3u8-get module spec")
m3u8_get = importlib.util.module_from_spec(spec)
if m3u8_get is None:
    raise ImportError("Failed to create m3u8-get module")
if spec.loader is None:
    raise ImportError("Failed to get loader for m3u8-get")
sys.modules["m3u8_get"] = m3u8_get
spec.loader.exec_module(m3u8_get)

# Export everything
__all__ = [
    "Track",
    "MasterPlaylist",
    "is_valid_m3u",
    "_get_base_url",
    "_normalize_url",
    "parse_m3u_attributes",
    "_parse_ext_x_media",
    "_create_video_track",
    "_add_associated_media_tracks",
    "_parse_ext_x_stream_inf",
    "_handle_standalone_media_tracks",
]

Track = m3u8_get.Track
MasterPlaylist = m3u8_get.MasterPlaylist
is_valid_m3u = m3u8_get.is_valid_m3u
_get_base_url = m3u8_get._get_base_url
_normalize_url = m3u8_get._normalize_url
parse_m3u_attributes = m3u8_get.parse_m3u_attributes
_parse_ext_x_media = m3u8_get._parse_ext_x_media
_create_video_track = m3u8_get._create_video_track
_add_associated_media_tracks = m3u8_get._add_associated_media_tracks
_parse_ext_x_stream_inf = m3u8_get._parse_ext_x_stream_inf
_handle_standalone_media_tracks = m3u8_get._handle_standalone_media_tracks
