# m3u8-get

Optimized asynchronous M3U8 downloader.

## Requirements

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) (recommended for installation)
- MKVToolNix (optional but highly recommended, for merging tracks)

## Usage

### Basic Usage

```bash
uv run python m3u8-get.py <MASTER_M3U_URL>
```

### With Custom Output Name

```bash
uv run python m3u8-get.py <MASTER_M3U_URL> <OUTPUT_NAME>
```

### Example

```bash
uv run python m3u8-get.py "https://example.com/playlist.m3u8" "my_video"
```

This will:

1. Fetch and parse the master M3U8 playlist
2. Display available tracks (video, audio, subtitles)
3. Let you interactively select the tracks you want
4. Download all selected tracks in parallel
5. Prompt you to merge tracks with FFmpeg

## Installation

### Using uv (recommended)

#### Clone the repository

```bash
git clone <repository-url>
cd m3u8-get
```

#### Install dependencies with uv

```bash
uv sync
```

## Configuration

Copy `.env.example` to `.env` and adjust the values:

```bash
cp .env.example .env
```

### Environment Variables

| Variable                   | Description                         | Default          |
| -------------------------- | ----------------------------------- | ---------------- |
| `MAX_CONCURRENT_DOWNLOADS` | Maximum parallel downloads          | `32`             |
| `CHUNK_SIZE`               | Chunk size for streaming (bytes)    | `1048576` (1 MB) |
| `TIMEOUT`                  | HTTP request timeout (seconds)      | `60`             |
| `RETRY_COUNT`              | Number of retry attempts on failure | `3`              |
| `MKVMERGE_PATH`            | Path to mkvmerge binary             | `mkvmerge`       |

### Performance tuning

- **Fast connection**: `MAX_CONCURRENT_DOWNLOADS=64`, `CHUNK_SIZE=4194304` (4 MB)
- **Standard connection**: `MAX_CONCURRENT_DOWNLOADS=32`, `CHUNK_SIZE=1048576` (1 MB)
- **Slow connection**: `MAX_CONCURRENT_DOWNLOADS=16`, `CHUNK_SIZE=1048576` (1 MB)

## Troubleshooting

### "mkvmerge not found"

Install MKVToolNix:

- **Windows**: `choco install mkvtoolnix` (Chocolatey) or download from [mkvtoolnix.download](https://mkvtoolnix.download/downloads.html)
- **macOS**: `brew install mkvtoolnix`
- **Linux**: `sudo apt install mkvtoolnix` (Debian/Ubuntu) or `sudo dnf install mkvtoolnix` (Fedora)

### Slow downloads

Increase `MAX_CONCURRENT_DOWNLOADS` in your `.env` file:

```.env
MAX_CONCURRENT_DOWNLOADS=64
```

### Download failures

- Check your internet connection
- Increase `TIMEOUT` and `RETRY_COUNT` in `.env`
- Some M3U8 playlists may require specific headers (edit `HEADERS` in the script)

## License

This project is provided as-is for educational purposes.
