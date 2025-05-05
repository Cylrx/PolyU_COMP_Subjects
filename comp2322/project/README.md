
# Multi-threaded HTTP Server

## Introduction

This project implements a multi-threaded HTTP server using low-level socket interfaces in Python. Unlike high-level server frameworks, this implementation builds HTTP functionality from scratch, allowing for a deeper understanding of network programming fundamentals and HTTP protocol mechanisms.

The server handles multiple client requests concurrently through a thread-based approach, supporting both text and image files, various HTTP status codes, and standard HTTP features like caching.

## Prerequisites

- Tested with Python 3.11.5
- No external libraries required beyond the Python standard library (
  - Optionally `pip install matplotlib` if you wish to run `test_thread_repeat.py`
  - Otherwise, just delete `test_thread_repeat.py` and there should not be an issue
- Compatible with Unix-like systems (Linux, macOS)
- **Note**: Windows is NOT officially supported due to differences in socket connection handling

## Installation

1. Clone or download this repository to your local machine
2. No additional installation steps required - the server runs with Python's standard library

## Directory Structure

```txt
project/
├── code/
│   ├── main.py          # Entry point and CLI handling
│   ├── server.py        # Core server functionality
│   ├── logger.py        # Logging implementation
│   ├── utils.py         # Helper utilities
│   ├── resources/       # Publicly accessible files
│   │   ├── index.html   # Landing page
│   │   ├── style.css    # Styling for landing page
│   │   ├── big.txt      # Large text file for testing
│   │   ├── cat.jpg      # Sample image
│   │   └── dog.webp     # Sample WebP image
│   ├── secrets/         # Non-accessible directory (for testing 403)
│   └── tests/           # Test scripts
│       ├── test_server.py      # Unit tests
│       ├── test_thread_repeat.py # Multi-threading tests (This takes a long time to run, not recommended)
│       └── client.py           # Test client
└── report/              # Project report and documentation
```

## Running the Server

### Basic Usage

```bash
cd <project-root-directory>  # I.e., the parent folder of code/ and report/
python3 code/main.py --arg1 <val1> --arg2 <val2>
```

### With Custom Parameters

```bash
python3 code/main.py --host 127.0.0.1 --port 8080 --timeout 15 --max-conn 100
```

### All Supported Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--host` | Host address to bind to | `127.0.0.1` |
| `--port` | Port number to listen on (if unavailable, will try 8080-65535) | `80` |
| `--max-requests` | Maximum number of requests handled in parallel | `1000` |
| `--max-threads` | Maximum number of threads in the thread pool | `100` |
| `--timeout` | Connection timeout in seconds | `15` |
| `--max-conn` | Maximum number of requests per persistent connection | `100` |
| `--cache-age` | Value for Cache-Control max-age (in seconds) | `86400` (24 hours) |
| `--log-file` | Path to log file | `server.log` |

## Features

### HTTP Protocol Support

- **HTTP Versions**: Supports HTTP/1.0 and HTTP/1.1
- **Methods**: Implements GET and HEAD methods
- **Status Codes**:
  - `200 OK`: Successful request
  - `304 Not Modified`: Resource not modified since client's cached version
  - `400 Bad Request`: Malformed request syntax
  - `403 Forbidden`: Access to requested resource is forbidden
  - `404 Not Found`: Requested resource does not exist
  - `415 Unsupported Media Type`: Server doesn't support the requested file type

### Connection Management

- **Persistent Connections**: Maintains connections based on HTTP version and headers
  - HTTP/1.1: Persistent by default (unless Connection: close is specified)
  - HTTP/1.0: Non-persistent by default (unless Connection: keep-alive is specified)
- **Connection Timeout**: Automatically closes idle connections after configured timeout

### Content Handling

- **File Types**: Supports common text and image files
  - Text: HTML, CSS, JavaScript, plain text, JSON, XML
  - Images: PNG, JPEG, GIF, ICO, SVG, WebP
- **Caching**: Implements Last-Modified and If-Modified-Since headers

### Security

- **Path Validation**: Prevents directory traversal attacks
- **Access Control**: Restricts access to only designated directories (/resources/)

### Logging

- **Request Logging**: Maintains a log file with client information, access time, requested file, and response type
- **Console Output**: Provides colored console output for monitoring server activity

## Testing the Server

### Browser Testing

1. Start the server as described in the "Running the Server" section
2. Open a web browser and navigate to: `http://127.0.0.1:[port]/`
3. The landing page provides links to test different server functionality:
   - Normal files (200 OK)
   - Non-existent files (404 Not Found)
   - Forbidden directories (403 Forbidden)
   - Unsupported file types (415 Unsupported Media Type)
   - Bad file paths (400 Bad Request)

### Running Unit Tests

1. Start the server with known configuration
2. In a separate terminal window, set environment variables matching your server configuration:

```bash
export TEST_SRVR_HOST="127.0.0.1"
export TEST_SRVR_PORT="8080"         # Must match server port
export TEST_SRVR_TIMEOUT="15"
export TEST_SRVR_MAX_CONN="100"
python3 -m unittest discover -v code/tests
```

**Important**: The environment variable values MUST match exactly the arguments used when starting the server.

## HTTP Response Headers

### For Successful Responses (200 OK)

- `Content-Length`: Size of the content in bytes
- `Last-Modified`: Last modification time of the file
- `Connection`: Connection type (keep-alive or close)
- `Keep-Alive`: Connection parameters (if applicable)
- `Cache-Control`: Caching directives

### For Error Responses

- Custom HTML error pages with relevant status code and message
- Appropriate headers indicating connection handling

## Troubleshooting

### Common Issues

- **Incorrect Port**: If the specified port is unavailable, the server automatically tries to find an available port starting from 8080. Check the console output for the actual port being used.
- **Connection Refused**: Ensure your firewall allows connections to the specified port.
- **File Access Errors**: Verify that the URL path starts with `/resources/`, which is the only directory accessible by the server.
- **Windows Compatibility**: The server may experience WinError 10054 and 10053 on Windows due to differences in connection handling.

### Debugging

- Check the server console for colored log messages
- Examine the `server.log` file for detailed request history
- For advanced debugging, use network analyzers like Wireshark to inspect HTTP traffic

## Limitations

- Only supports HTTP/1.0 and HTTP/1.1 (not HTTP/2 or HTTP/3)
- Limited to GET and HEAD methods (no POST, PUT, DELETE, etc.)
- Does not support SSL/TLS (HTTPS)
- Windows compatibility is not guaranteed
- Content-types are limited to common text and image formats
