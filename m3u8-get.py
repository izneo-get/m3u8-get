# -*- coding: utf-8 -*-
__version__ = "0.1.0"
"""
m3u8-get.py - Optimized asynchronous M3U8 downloader

Async version of m3u8-get.py with:
- Parallel downloads via aiohttp
- tqdm progress bar
- Configuration via .env
- Error handling with retry logic
- Interactive track selection with questionary
"""

import asyncio
import heapq
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
import questionary
import questionary.constants
from aiohttp import ClientTimeout, TCPConnector
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load environment variables from .env if present
load_dotenv()

# Default values
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "32"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", str(1024 * 1024)))  # 1 MB default
TIMEOUT = int(os.getenv("TIMEOUT", "60"))  # 60 seconds default
RETRY_COUNT = int(os.getenv("RETRY_COUNT", "3"))  # 3 attempts default
MKVMERGE_PATH = os.getenv("MKVMERGE_PATH", "mkvmerge")  # Path to mkvmerge binary

# Custom DNS servers (space-separated)
DNS_SERVERS_STR = os.getenv("DNS_SERVERS", "").strip()
DNS_SERVERS = DNS_SERVERS_STR.split() if DNS_SERVERS_STR else None

# HTTP headers (same as original)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) Gecko/20100101 Firefox/138.0",
    "Accept": "*/*",
    "Accept-Language": "fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3",
    "Origin": "https://rapid-cloud.co",
    "Connection": "keep-alive",
    "Referer": "https://rapid-cloud.co/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "cross-site",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
}


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class Track:
    """Represents a media track (video, audio, or subtitle)."""

    type: str  # 'video', 'audio', 'subtitle'
    name: str
    language: str = ""
    group_id: str = ""
    url: str = ""
    bandwidth: int = 0
    resolution: str = ""
    channels: str = ""
    is_default: bool = False
    codec: str = ""
    index: int = 0

    def __str__(self):
        """String representation for display in questionary."""
        if self.type == "video":
            res = f" ({self.resolution})" if self.resolution else ""
            bw = f" @ {self.bandwidth // 1000000}M" if self.bandwidth else ""
            return f"🎬 Video{res}{bw}"
        elif self.type == "audio":
            lang = f" [{self.language}]" if self.language else ""
            ch = f" {self.channels}ch" if self.channels else ""
            default = " [DEFAULT]" if self.is_default else ""
            return f"🔊 Audio{lang}{ch}{default}"
        else:  # subtitle
            lang = f" [{self.language}]" if self.language else ""
            default = " [DEFAULT]" if self.is_default else ""
            return f"📝 Subtitle{lang}{default}"


@dataclass
class MasterPlaylist:
    """Represents a parsed master M3U8 playlist."""

    tracks: list[Track] = field(default_factory=list)
    base_url: str = ""

    def get_tracks_by_type(self, track_type: str) -> list[Track]:
        """Get all tracks of a specific type."""
        return [t for t in self.tracks if t.type == track_type]


# ============================================================================
# DISPLAY FUNCTIONS AND UTILITIES
# ============================================================================


def print_banner() -> None:
    """Display the program banner."""
    print(f"\n{'='*60}")
    print(
        """
███╗   ███╗██████╗ ██╗   ██╗ █████╗ 
████╗ ████║╚════██╗██║   ██║██╔══██╗
██╔████╔██║ █████╔╝██║   ██║╚█████╔╝
██║╚██╔╝██║ ╚═══██╗██║   ██║██╔══██╗
██║ ╚═╝ ██║██████╔╝╚██████╔╝╚█████╔╝
╚═╝     ╚═╝╚═════╝  ╚═════╝  ╚════╝ 
                                    
       ██████╗ ███████╗████████╗    
      ██╔════╝ ██╔════╝╚══██╔══╝    
█████╗██║  ███╗█████╗     ██║       
╚════╝██║   ██║██╔══╝     ██║       
      ╚██████╔╝███████╗   ██║       
       ╚═════╝ ╚══════╝   ╚═╝        
"""
    )
    print(f"  v{__version__}")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Parallel downloads: {MAX_CONCURRENT_DOWNLOADS}")
    print(f"  - Chunk size: {CHUNK_SIZE // 1024} KB")
    print(f"  - Timeout: {TIMEOUT}s")
    print(f"  - Retry count: {RETRY_COUNT}")
    print(f"{'='*60}\n")


def print_usage() -> None:
    """Display program usage help."""
    print("Usage: ")
    print(f"{sys.argv[0]} <MASTER_M3U_URL> [OUTPUT_NAME]")


def print_track_summary(tracks: List[Track]) -> None:
    """Print summary of available tracks."""
    video = [t for t in tracks if t.type == "video"]
    audio = [t for t in tracks if t.type == "audio"]
    subs = [t for t in tracks if t.type == "subtitle"]

    print(f"\n📋 Available tracks:")
    print(f"   🎬 Video: {len(video)}")
    print(f"   🔊 Audio: {len(audio)}")
    print(f"   📝 Subtitles: {len(subs)}")


def create_dns_resolver() -> Optional[aiohttp.AsyncResolver]:
    """
    Create a custom DNS resolver if DNS_SERVERS is configured.

    Returns:
        AsyncResolver if DNS servers are configured, None otherwise
    """
    if DNS_SERVERS:
        try:
            resolver = aiohttp.AsyncResolver(nameservers=DNS_SERVERS)
            print(f"🌐 Using custom DNS servers: {', '.join(DNS_SERVERS)}")
            return resolver
        except Exception as e:
            print(f"⚠️  Failed to create custom DNS resolver: {e}")
            print(f"   Falling back to system DNS")
            return None
    return None


# ============================================================================
# M3U8 PARSING
# ============================================================================


async def parse_master_m3u(session: aiohttp.ClientSession, master_m3u_url: str) -> MasterPlaylist:
    """
    Parse a master M3U8 playlist and extract all tracks.

    Args:
        session: aiohttp session
        master_m3u_url: Master playlist URL

    Returns:
        MasterPlaylist object with all tracks
    """
    timeout = ClientTimeout(total=TIMEOUT)
    async with session.get(master_m3u_url, headers=HEADERS, timeout=timeout) as resp:
        content = await resp.text()

    playlist = MasterPlaylist()
    playlist.base_url = "/".join(master_m3u_url.split("/")[:-1]) + "/"

    lines = content.split("\n")
    i = 0

    # First pass: collect all EXT-X-MEDIA entries (audio and subtitles)
    media_tracks = {}
    current_header = {}

    for i, line in enumerate(lines):
        line = line.strip()

        # Parse EXT-X-MEDIA (audio and subtitles)
        if line.startswith("#EXT-X-MEDIA:"):
            attrs = parse_m3u_attributes(line)

            track_type = attrs.get("TYPE", "").lower()
            if track_type in ("audio", "subtitles"):
                track = Track(
                    type="audio" if track_type == "audio" else "subtitle",
                    name=attrs.get("NAME", "Unknown"),
                    language=attrs.get("LANGUAGE", attrs.get("NAME", "")),
                    group_id=attrs.get("GROUP-ID", ""),
                    url=attrs.get("URI", ""),
                    channels=attrs.get("CHANNELS", ""),
                    is_default=attrs.get("DEFAULT", "NO") == "YES",
                    codec=attrs.get("CODECS", ""),
                )

                # Store with GROUP-ID as key for reference by video streams
                group_id = attrs.get("GROUP-ID", "")
                if group_id:
                    if group_id not in media_tracks:
                        media_tracks[group_id] = []
                    media_tracks[group_id].append(track)

    # Second pass: parse EXT-X-STREAM-INF (video tracks)
    video_index = 0
    for i, line in enumerate(lines):
        line = line.strip()

        if line.startswith("#EXT-X-STREAM-INF:"):
            attrs = parse_m3u_attributes(line)

            # Next line should be the URL
            if i + 1 < len(lines):
                url = lines[i + 1].strip()
                if not url or url.startswith("#"):
                    continue

                # Convert to absolute URL if needed
                if not url.startswith("http"):
                    url = urljoin(playlist.base_url, url)

                resolution = attrs.get("RESOLUTION", "")
                bandwidth = int(attrs.get("BANDWIDTH") or "0")
                codecs = attrs.get("CODECS", "")
                audio_group = attrs.get("AUDIO", "")
                subs_group = attrs.get("SUBTITLES", "")

                # Create video track
                video_track = Track(
                    type="video",
                    name=f"Stream {video_index + 1}",
                    resolution=resolution,
                    bandwidth=bandwidth,
                    codec=codecs,
                    url=url,
                    index=video_index,
                )
                playlist.tracks.append(video_track)

                # Add associated audio tracks
                if audio_group and audio_group in media_tracks:
                    for audio_track in media_tracks[audio_group]:
                        # Convert to absolute URL if needed
                        audio_url = audio_track.url
                        if audio_url and not audio_url.startswith("http"):
                            audio_url = urljoin(playlist.base_url, audio_url)
                        audio_track.url = audio_url
                        if audio_track not in playlist.tracks:
                            playlist.tracks.append(audio_track)

                # Add associated subtitle tracks
                if subs_group and subs_group in media_tracks:
                    for sub_track in media_tracks[subs_group]:
                        # Convert to absolute URL if needed
                        sub_url = sub_track.url
                        if sub_url and not sub_url.startswith("http"):
                            sub_url = urljoin(playlist.base_url, sub_url)
                        sub_track.url = sub_url
                        if sub_track not in playlist.tracks:
                            playlist.tracks.append(sub_track)

                video_index += 1

    # If no video streams found, check for standalone media tracks
    if video_index == 0 and media_tracks:
        for group_tracks in media_tracks.values():
            for track in group_tracks:
                if track.url and not track.url.startswith("http"):
                    track.url = urljoin(playlist.base_url, track.url)
                if track not in playlist.tracks:
                    playlist.tracks.append(track)

    return playlist


def parse_m3u_attributes(line: str) -> Dict[str, str]:
    """
    Parse attributes from an M3U8 tag line.

    Args:
        line: M3U8 tag line (e.g., #EXT-X-STREAM-INF:BANDWIDTH=1000,RESOLUTION=1920x1080)

    Returns:
        Dictionary of attribute names and values
    """
    # Remove the tag prefix
    if ":" in line:
        line = line.split(":", 1)[1]

    attrs = {}
    # Pattern to match KEY="VALUE" or KEY=VALUE
    # This handles commas inside quoted strings correctly
    pattern = r'([A-Z-]+)="([^"]*)"|([A-Z-]+)=([^,]+)'

    for match in re.finditer(pattern, line):
        if match.group(1):  # Quoted value case: KEY="VALUE"
            key = match.group(1)
            value = match.group(2)
        else:  # Unquoted value case: KEY=VALUE
            key = match.group(3)
            value = match.group(4)

        if key and value is not None:
            # Remove escape characters
            value = value.replace(r"\"", '"').replace(r"\\", "\\")
            attrs[key] = value

    return attrs


# ============================================================================
# ASYNC DOWNLOAD FUNCTIONS
# ============================================================================


async def fetch_with_retry(
    session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore, retry_count: int = RETRY_COUNT
) -> bytes:
    """
    Download a URL with retry handling and exponential backoff.

    Args:
        session: aiohttp session
        url: URL to download
        semaphore: Semaphore to limit concurrency
        retry_count: Number of attempts

    Returns:
        Downloaded binary content

    Raises:
        aiohttp.ClientError: If error occurs after all attempts
    """
    base_delay = 0.5  # 500ms initial delay
    last_error = None

    for attempt in range(retry_count):
        async with semaphore:
            try:
                timeout = ClientTimeout(total=TIMEOUT)
                async with session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"HTTP {response.status}",
                        )

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < retry_count - 1:
                    # Exponential backoff
                    delay = base_delay * (2**attempt)
                    await asyncio.sleep(delay)
                else:
                    raise

    # This should never be reached, but satisfies the type checker
    raise aiohttp.ClientError(f"Failed to fetch {url} after {retry_count} attempts")


async def download_segment(
    session: aiohttp.ClientSession, url: str, index: int, semaphore: asyncio.Semaphore, progress_bar: tqdm
) -> tuple[int, bytes]:
    """
    Download a segment asynchronously.

    Args:
        session: aiohttp session
        url: Segment URL
        index: Segment index (for ordering)
        semaphore: Semaphore to limit concurrency
        progress_bar: tqdm progress bar

    Returns:
        Tuple (index, data) with the index and downloaded data
    """
    try:
        data = await fetch_with_retry(session, url, semaphore)
        progress_bar.update(1)
        return (index, data)
    except Exception as e:
        print(f"\n[ERROR] Failed to download segment {index}: {e}")
        # Return empty data to not block the entire download
        return (index, b"")


async def download_track(session: aiohttp.ClientSession, track: Track, output_folder: str) -> Optional[str]:
    """
    Download a complete track (video, audio, or subtitle) with streaming write.

    Uses a heapq-based writer that writes segments to disk as soon as they
    are available in the correct order, minimizing memory usage.

    Args:
        session: aiohttp session
        track: Track to download
        output_folder: Output folder path

    Returns:
        Path to downloaded file, or None on failure
    """
    try:
        # First, fetch the track's playlist to get segments
        timeout: ClientTimeout = ClientTimeout(total=TIMEOUT)
        async with session.get(track.url, headers=HEADERS, timeout=timeout) as resp:
            content: str = await resp.text()

        # Parse segments from playlist
        lines: List[str] = content.split("\n")
        segment_urls: List[str] = [line.strip() for line in lines if line.strip() and not line.startswith("#")]

        if not segment_urls:
            print(f"[WARNING] No segments found for {track}")
            return None

        # Determine base URL for segments
        base_url: str = urljoin(track.url, ".")

        # Convert relative URLs to absolute
        absolute_urls: List[str] = []
        for url in segment_urls:
            if not url.startswith("http://") and not url.startswith("https://"):
                absolute_urls.append(f"{base_url}/{url}")
            else:
                absolute_urls.append(url)

        # Determine filename
        ext: str = get_track_extension(track, absolute_urls[0])
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename: str = f"{timestamp}_{track.type}"

        if track.type == "video" and track.resolution:
            filename += f"_{track.resolution.replace('x', 'p')}"
        if track.language:
            filename += f"_{track.language}"

        filename += f".{ext}"
        output_path: str = f"{output_folder}/{filename}"

        # Download all segments
        total_segments: int = len(absolute_urls)

        # Create semaphore to limit concurrency
        semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

        # Shared structures for streaming write
        segment_heap: List[Tuple[int, bytes]] = []
        heap_lock: asyncio.Lock = asyncio.Lock()
        write_lock: asyncio.Lock = asyncio.Lock()  # Prevents concurrent writes
        next_to_write: int = 0

        # File opened once, shared with writer function
        output_file: Any = open(output_path, "wb")

        async def write_all_available() -> None:
            """Write all consecutive segments from the heap."""
            nonlocal next_to_write
            # Prevent concurrent writes to file
            async with write_lock:
                async with heap_lock:
                    # Write all consecutive segments that are available
                    while segment_heap and segment_heap[0][0] == next_to_write:
                        index: int
                        data: bytes
                        index, data = heapq.heappop(segment_heap)
                        output_file.write(data)
                        next_to_write += 1

        # Download coroutine - downloads and writes available segments
        async def download_and_enqueue(url: str, index: int) -> None:
            try:
                data: bytes = await fetch_with_retry(session, url, semaphore)
                async with heap_lock:
                    heapq.heappush(segment_heap, (index, data))
                download_progress.update(1)
                # Try to write all available segments
                await write_all_available()
            except Exception as e:
                print(f"\n[ERROR] Failed to download segment {index}: {e}")
                # Push empty data to not block the writer
                async with heap_lock:
                    heapq.heappush(segment_heap, (index, b""))
                download_progress.update(1)
                await write_all_available()

        # Create progress bar for download
        with tqdm(
            total=total_segments,
            desc=f"[{track.type.upper()}] {track.name}",
            unit="seg",
            ncols=80,
            colour="green",
        ) as download_progress:

            # Create all download tasks
            start_time: float = time.time()
            download_tasks = [download_and_enqueue(url, i) for i, url in enumerate(absolute_urls)]

            # Execute all download tasks in parallel
            await asyncio.gather(*download_tasks, return_exceptions=True)

            # After all downloads are done, write any remaining segments
            await write_all_available()

            elapsed_time: float = time.time() - start_time

        # Close the output file
        output_file.close()

        # Display statistics
        file_size: float = os.path.getsize(output_path) / (1024 * 1024)  # MB
        speed: float = file_size / elapsed_time if elapsed_time > 0 else 0

        print(f"  ✓ {track}: {file_size:.2f} MB in {elapsed_time:.1f}s ({speed:.2f} MB/s)")

        return output_path

    except Exception as e:
        print(f"\n[ERROR] Failed to download {track}: {e}")
        return None
    finally:
        # Always close the output file
        output_file = locals().get("output_file")
        if output_file is not None:
            output_file.close()


def get_track_extension(track: Track, first_segment_url: str) -> str:
    """Determine file extension for a track."""
    if track.type == "subtitle":
        return "vtt"  # WebVTT for subtitles
    else:
        # Extract from segment URL
        url_ext = first_segment_url.split("?")[0].split(".")[-1]
        return url_ext if url_ext in ("ts", "m4s", "mp4") else "ts"


# ============================================================================
# INTERACTIVE SELECTION
# ============================================================================


def select_tracks_interactive(playlist: MasterPlaylist) -> List[Track]:
    """
    Interactive track selection using questionary.

    Args:
        playlist: Parsed master playlist

    Returns:
        List of selected tracks
    """
    # Group tracks by type
    video_tracks: List[Track] = playlist.get_tracks_by_type("video")
    audio_tracks: List[Track] = playlist.get_tracks_by_type("audio")
    sub_tracks: List[Track] = playlist.get_tracks_by_type("subtitle")

    # If only one video track, auto-select it
    selected_tracks: List[Track] = []

    if len(video_tracks) == 1:
        selected_tracks.append(video_tracks[0])
        print(f"Auto-selected: {video_tracks[0]}")
    elif len(video_tracks) > 1:
        # Let user choose video track(s)
        choices: List[questionary.Choice] = [questionary.Choice(str(t), value=t) for t in video_tracks]
        answer: Optional[List[Track]] = questionary.checkbox(
            "🎬 Select video track(s):",
            choices=choices,
            validate=lambda x: len(x) > 0 or "You must select at least one video track",
        ).ask()

        if answer is None:  # User cancelled
            return []
        selected_tracks.extend(answer)

    # Audio selection
    if audio_tracks:
        choices: List[questionary.Choice] = [questionary.Choice(str(t), value=t) for t in audio_tracks]
        answer: Optional[List[Track]] = questionary.checkbox(
            "🔊 Select audio track(s):",
            choices=choices,
            instruction="(press SPACE to select, ENTER to confirm)",
        ).ask()

        if answer is not None:
            selected_tracks.extend(answer)

    # Subtitle selection
    if sub_tracks:
        choices: List[questionary.Choice] = [questionary.Choice(str(t), value=t) for t in sub_tracks]
        answer: Optional[List[Track]] = questionary.checkbox(
            "📝 Select subtitle track(s):",
            choices=choices,
            instruction="(press SPACE to select, ENTER to confirm, or skip with ENTER)",
        ).ask()

        if answer is not None:
            selected_tracks.extend(answer)

    return selected_tracks


# ============================================================================
# MAIN ASYNC FUNCTION
# ============================================================================


async def fetch_and_parse_playlist(master_m3u_url: str) -> MasterPlaylist:
    """
    Fetch and parse the master M3U8 playlist (async).

    Args:
        master_m3u_url: Master M3U playlist URL

    Returns:
        Parsed MasterPlaylist object
    """
    # Create custom DNS resolver if configured
    resolver = create_dns_resolver()

    # Configure connector for the session with custom DNS if available
    if resolver:
        connector: TCPConnector = TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            keepalive_timeout=30,
            resolver=resolver,
        )
    else:
        connector = TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            keepalive_timeout=30,
        )

    timeout_config: ClientTimeout = ClientTimeout(total=TIMEOUT)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout_config, headers=HEADERS) as session:
        playlist: MasterPlaylist = await parse_master_m3u(session, master_m3u_url)
        return playlist


async def download_selected_tracks(selected_tracks: List[Track], output_folder: str, file_out_name: str) -> List[str]:
    """
    Download all selected tracks (async).

    Args:
        selected_tracks: List of tracks to download
        output_folder: Output folder path
        file_out_name: Output filename prefix

    Returns:
        List of downloaded file paths
    """
    # Create custom DNS resolver if configured
    resolver = create_dns_resolver()

    # Configure connector for the session with custom DNS if available
    if resolver:
        connector: TCPConnector = TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            keepalive_timeout=30,
            resolver=resolver,
        )
    else:
        connector = TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            keepalive_timeout=30,
        )

    timeout_config: ClientTimeout = ClientTimeout(total=TIMEOUT)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout_config, headers=HEADERS) as session:
        # Download all selected tracks
        print(f"\n⬇️  Downloading {len(selected_tracks)} track(s)...\n")

        downloaded_files: List[str] = []
        for track in selected_tracks:
            file_path: Optional[str] = await download_track(session, track, output_folder)
            if file_path:
                downloaded_files.append(file_path)

        # Step 4: Display summary
        print(f"\n{'='*60}")
        print(f"Download completed!")
        print(f"  - Tracks downloaded: {len(downloaded_files)}/{len(selected_tracks)}")
        print(f"  - Output folder: {output_folder}")
        print(f"{'='*60}")

        if downloaded_files:
            print(f"\n📁 Downloaded files:")
            for f in downloaded_files:
                print(f"   - {f}")
        else:
            print("\n⚠️  No files were successfully downloaded.")

        return downloaded_files


def build_mkvmerge_command(
    downloaded_files: List[str], output_folder: str, file_out_name: str
) -> Tuple[List[str], str]:
    """
    Build mkvmerge command to merge downloaded tracks.

    Args:
        downloaded_files: List of downloaded file paths
        output_folder: Output folder path
        file_out_name: Output filename prefix

    Returns:
        Tuple of (mkvmerge command as list, output path)
    """
    output_name: str = file_out_name if file_out_name else "output"
    output_path: str = f"{output_folder}/{output_name}.mkv"

    # Build mkvmerge command
    mkvmerge_cmd: List[str] = [MKVMERGE_PATH, "-o", output_path]
    mkvmerge_cmd.extend(downloaded_files)

    return mkvmerge_cmd, output_path


def prompt_and_run_mkvmerge(downloaded_files: List[str], output_folder: str, file_out_name: str) -> None:
    """
    Prompt user and run mkvmerge to merge tracks (synchronous).

    Args:
        downloaded_files: List of downloaded file paths
        output_folder: Output folder path
        file_out_name: Output filename prefix
    """
    mkvmerge_cmd: List[str]
    output_path: str
    mkvmerge_cmd, output_path = build_mkvmerge_command(downloaded_files, output_folder, file_out_name)

    # Display the command
    cmd_display: str = " ".join([f'"{x}"' if " " in x else x for x in mkvmerge_cmd])
    print(f"\n🎬 MKVMerge command to remux tracks:")
    print(f"   {cmd_display}")

    # Ask if user wants to run it (outside of async context)
    run_mkvmerge: Optional[bool] = questionary.confirm(
        "Would you like me to run this MKVMerge command now?", default=True
    ).ask()

    if run_mkvmerge:
        print(f"\n🎬 Running MKVMerge to merge tracks...\n")
        # Run synchronously
        result: subprocess.CompletedProcess[bytes] = subprocess.run(mkvmerge_cmd, check=False)
        if result.returncode == 0:
            print(f"\n✅ Merge completed: {output_path}")

            # Ask if user wants to delete intermediate files
            delete_intermediates: Optional[bool] = questionary.confirm(
                "Would you like to delete the intermediate files?", default=True
            ).ask()

            if delete_intermediates:
                print(f"\n🗑️  Deleting intermediate files...")
                for file_path in downloaded_files:
                    try:
                        os.remove(file_path)
                        print(f"   - Deleted: {file_path}")
                    except OSError as e:
                        print(f"   - Failed to delete {file_path}: {e}")
                print("✅ Cleanup completed!")
        else:
            print(f"\n⚠️  MKVMerge exited with code {result.returncode}")


# ============================================================================
# ENTRY POINT
# ============================================================================


def main() -> None:
    """Program entry point."""
    # Check arguments
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    file_out_name: str = sys.argv[2] if len(sys.argv) > 2 else ""
    master_m3u_url: str = sys.argv[1]

    # Display banner
    print_banner()

    # Create tmp folder if necessary
    output_folder: str = f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/tmp"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    try:
        # Step 1: Fetch and parse playlist (async)
        playlist: MasterPlaylist = asyncio.run(fetch_and_parse_playlist(master_m3u_url))

        print_track_summary(playlist.tracks)

        if not playlist.tracks:
            print("\n[ERROR] No tracks found in playlist!")
            return

        # Step 2: Interactive track selection (synchronous - questionary)
        selected_tracks: List[Track] = select_tracks_interactive(playlist)

        if not selected_tracks:
            print("\nNo tracks selected. Exiting.")
            return

        print(f"\n🎯 Selected {len(selected_tracks)} track(s):")
        for track in selected_tracks:
            print(f"   - {track}")

        # Step 3: Download tracks (async)
        downloaded_files: List[str] = asyncio.run(
            download_selected_tracks(selected_tracks, output_folder, file_out_name)
        )

        # Step 4: Prompt for MKVMerge merge (synchronous)
        if downloaded_files:
            prompt_and_run_mkvmerge(downloaded_files, output_folder, file_out_name)

    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
