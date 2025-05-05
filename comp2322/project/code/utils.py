import os
import socket
import urllib.parse
from pathlib import Path
from logger import log_err, log_warn
import re
import time

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_MAP = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def check_bind(host: str, port: int) -> bool:
    if port > 65535 or port < 0: 
        log_err(f"[!] Invalid port: {port}")
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.bind((host, port))
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            log_warn(f'[!] port {port} is already in use')
            return False


def recv_req(c_sock: socket.socket, timeout: int) -> tuple[str, bool]:
    req_buffer = b""
    c_sock.settimeout(timeout)
    try:
        while b"\r\n\r\n" not in req_buffer:
            data = c_sock.recv(1)
            if not data:
                return "", True
            req_buffer += data
        
        req_str = req_buffer.decode('utf-8')
        return req_str, True
    except socket.timeout:
        raise socket.timeout
    except Exception as e:
        log_err(f"[!] Unkown Error: {e}")
        return "", False 

def valid_path(path: str) -> bool:
    if not path.startswith('/') or '..' in path or '//' in path:
        return False
    
    bad_chars = ['\x00', '\n', '\r', '\t', '\v', '\f']
    if any(char in path for char in bad_chars):
        return False
    
    try:
        decoded = urllib.parse.unquote(path)
        if '..' in decoded:
            return False
    except Exception:
        return False
    
    if len(path) > 1024:
        return False
        
    return True


def get_path(path: str) -> str:
    if path == '/':
        path = '/resources/index.html'
    if path == '/favicon.ico': 
        path = '/resources/favicon.ico'

    root = Path(__file__).resolve().parent
    path = urllib.parse.unquote(path).lstrip('/')
    full_path = os.path.normpath(os.path.join(root, path))
    return path, full_path


def parse_http(req_data: str) -> dict:
    lines = req_data.strip().split("\r\n")
    method, path, version = lines[0].split()
    hdrs = {}

    for line in lines[1:]:
        if line:
            key, value = line.split(": ", 1)
            hdrs[key.strip().lower()] = value.strip()

    return {
        "method": method,
        "path": path,
        "version": version,
        "headers": hdrs,
    }


def get_content(full_path: str) -> bytes:
    try:
        with open(full_path, 'rb') as f:
            return f.read()
    except Exception as e:
        log_err(f"[!] Error reading file: {e}")
        return b""


def get_mod_time(full_path: str) -> str:
    """
    Return the modification time of the file
    Similar format as If-Modified-Since header
    """
    timestamp = os.path.getmtime(full_path)
    gmtime = time.gmtime(timestamp)
    
    day_name = DAYS[gmtime.tm_wday]
    day = f"{gmtime.tm_mday:02d}"
    month = MONTHS[gmtime.tm_mon - 1]
    year = str(gmtime.tm_year)
    hour = f"{gmtime.tm_hour:02d}"
    minute = f"{gmtime.tm_min:02d}"
    second = f"{gmtime.tm_sec:02d}"
    
    formatted_time = f"{day_name}, {day} {month} {year} {hour}:{minute}:{second} GMT"
    return formatted_time


def valid_mod_time(mod_time: str) -> bool:
    """
    Check If-Modified-Since header's format
    
    Valid format:
        If-Modified-Since: <day-name>, <day> <month> <year> <hour>:<minute>:<second> GMT
    
    Directives
        <day-name> - One of Mon, Tue, Wed, Thu, Fri, Sat, or Sun (case-sensitive).
        <day> - 2 digit day number, e.g., "04" or "23".
        <month> - One of Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec (case sensitive).
        <year> - 4 digit year number, e.g., "1990" or "2016".
        <hour> - 2 digit hour number, e.g., "09" or "23".
        <minute> - 2 digit minute number, e.g., "04" or "59".
        <second> - 2 digit second number, e.g., "04" or "59".
        GMT - Greenwich Mean Time. HTTP dates are always expressed in GMT, never in local time.
    
    Example: 
        If-Modified-Since: Wed, 21 Oct 2015 07:28:00 GMT
    """
    pattern = r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d{2}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{4}) (\d{2}):(\d{2}):(\d{2}) GMT$"
    return bool(re.match(pattern, mod_time))


def cmp_mod_time(if_mod_since: str, last_mod: str) -> bool:
    """Compare If-Modified-Since date with Last-Modified date
    Return True if the file has not been modified since the date in if_mod_since

    Both last_mod and if_mod_since are of same format
    """
    expr = r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d{2}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{4}) (\d{2}):(\d{2}):(\d{2}) GMT$"

    # if_mod_since
    match = re.match(expr, if_mod_since)
    if not match: return False
    _, day1, month1, year1, hour1, min1, sec1 = match.groups()
    
    # last_mod
    match = re.match(expr, last_mod)
    if not match: return False
    _, day2, month2, year2, hour2, min2, sec2 = match.groups()
    
    if int(year1) > int(year2): return True
    if int(year1) < int(year2): return False
    
    month_idx1 = MONTHS.index(month1)
    month_idx2 = MONTHS.index(month2)
    if month_idx1 > month_idx2: return True
    if month_idx1 < month_idx2: return False
    
    if int(day1) > int(day2): return True
    if int(day1) < int(day2): return False
    
    if int(hour1) > int(hour2): return True
    if int(hour1) < int(hour2): return False
    
    if int(min1) > int(min2): return True
    if int(min1) < int(min2): return False
    
    return int(sec1) >= int(sec2)


def allowed_path(full_path: str) -> bool:
    """
    Only accept paths inside the /resources/ directory.
    Handles both Windows and POSIX path formats.
    """
    try:
        root = Path(__file__).resolve().parent
        full_path_obj = Path(full_path).resolve()
        relative_path = full_path_obj.relative_to(root)

        # Convert to POSIX style ('/' separators) and check prefix
        # Check starts with 'resources/' (no leading slash needed after relative_to)
        return relative_path.as_posix().startswith('resources/')
    except ValueError:
        return False
    except Exception:
        log_err(f"[!] Error processing path in allowed_path: {full_path}")
        return False
    