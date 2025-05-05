import socket
import os
import utils
import concurrent

from logger import log_info, log_warn, log_err, log_succ, FileLogger

# only support text or images
CONTENT_TYPES = {
    'html': 'text/html',
    'css': 'text/css',
    'js': 'text/javascript',
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'gif': 'image/gif',
    'ico': 'image/x-icon',
    'svg': 'image/svg+xml',
    'webp': 'image/webp',
    'txt': 'text/plain',
    'json': 'application/json',
    'xml': 'application/xml',
}

_POOL = None

def init_pool(workers: int):
    global _POOL
    _POOL = concurrent.futures.ThreadPoolExecutor(max_workers=workers)

def shutdown_pool():
    global _POOL
    _POOL.shutdown(wait=False)

def handle_cli(
        c_sock: socket.socket, 
        c_addr: tuple[str, int], 
        logger: FileLogger, 
        timeout: int,
        max_conn: int,
        cache_age: int
    ) -> None:
    try:
        log_info(f'Received connection from {c_addr}')
        _POOL.submit(run_cli, c_sock, c_addr, logger, timeout, max_conn, cache_age)
    except Exception as e:
        log_err(f"[!] Error - {e}")


def run_cli(
        c_sock: socket.socket, 
        c_addr: tuple[str, int], 
        logger: FileLogger, 
        timeout: int,
        max_conn: int,
        cache_age: int
    ) -> None:
    """
    Returns: 
        host: str - host address
        port: int - port number
        path: str - requested path by client
        code: int - server's response code
    """
    try: 
        keep_alive = True
        while keep_alive and max_conn > 0:
            req_data, recv_ok = utils.recv_req(c_sock, timeout)
            if (recv_ok and req_data): 
                keep_alive = handle_req(c_sock, c_addr, logger, req_data, timeout, max_conn, cache_age)
            elif (recv_ok and not req_data):  
                log_info(f'[*] Connection closed by remote {c_addr}')
                return
            else:
                send(c_sock, build(400, {}, b'', 'GET'))
                logger.entry(c_addr[0], c_addr[1], 'BAD-REQUEST', 400)
                return
            max_conn -= 1

    except socket.timeout:
        log_info(f'[*] Connection timeout: {c_addr}')
        return
    finally:
        try: 
            c_sock.close()
            log_info(f'[*] Connection closed gracefully: {c_addr}')
        except Exception as e:
            log_err(f'[*] Error closing connection: {e}')


def handle_req(
        c_sock: socket.socket, 
        c_addr: tuple[str, int], 
        logger: FileLogger, 
        req_data: str, 
        timeout: int,
        max_conn: int,
        cache_age: int
    ) -> bool:

    try: 
        http_req = utils.parse_http(req_data)
        if not http_req:
            raise ValueError
    except: 
        send(c_sock, build(400, {}, b'', 'GET'))
        logger.entry(c_addr[0], c_addr[1], 'BAD-REQUEST', 400)
        return False
    
    method = http_req['method'].upper()
    path = http_req['path']
    ver = http_req['version']
    hdrs = http_req['headers']
    

    def sender(status_code, log_msg, path_info='BAD-REQUEST', hdrs={}, content=b'', log_type='warn'):
        
        if keep_alive: 
            hdrs['Connection'] = 'keep-alive'
            hdrs['Keep-Alive'] = f'timeout={timeout}, max={max_conn}'
        else:
            hdrs['Connection'] = 'close'
        send(c_sock, build(status_code, hdrs, content, method, path_info))
        match log_type:
            case 'warn': log_warn(f'[!] {log_msg}')
            case 'err': log_err(f'[!] {log_msg}')
            case 'info': log_info(f'[*] {log_msg}')

        logger.entry(c_addr[0], c_addr[1], path_info, status_code)
        return keep_alive

    
    keep_alive = False
    
    if ver not in ['HTTP/1.0', 'HTTP/1.1']:
        return sender(400, f'Unsupported version: {ver}')

    # keep_alive rules: 
    # if HTTP/1.1, connection != 'close' or is empty, keep_alive is true
    # if HTTP/1.1, connection == 'close', keep_alive is false
    # if HTTP/1.0, connection != 'keep-alive' or is empty, keep_alive is false
    # if HTTP/1.0, connection == 'keep-alive', keep_alive is true

    # Intuitively, HTTP/1.1 should be kept alive by default, unless explicitly set connection = 'close'
    # HTTP/1.0 should be closed by default, unless explicitly set connection = 'keep-alive'

    keep_alive = ver == 'HTTP/1.1'
    if 'connection' in hdrs:
        conn_val = hdrs['connection'].lower()
        keep_alive = conn_val == 'keep-alive' if ver == 'HTTP/1.0' else conn_val != 'close'

    if method not in ['GET', 'HEAD']:
        return sender(400, f'Unsupported method: {method}')
    
    if not utils.valid_path(path):
        return sender(400, f'Invalid path: {path}')
    
    path, full_path = utils.get_path(path)
    
    if not utils.allowed_path(full_path):
        return sender(403, f'Access denied: {path}', path)

    if not os.path.exists(full_path) or os.path.isdir(full_path):
        return sender(404, f'File not found: {path}', path)
    
    ext = os.path.splitext(full_path)[1].lstrip('.').lower()
    if ext not in CONTENT_TYPES:
        return sender(415, f'Unsupported content type: {ext}', path)
    
    if_mod_since = hdrs.get('if-modified-since') 
    if if_mod_since and not utils.valid_mod_time(if_mod_since):
        return sender(400, f'Invalid if-modified-since: {if_mod_since}', path)

    last_mod = utils.get_mod_time(full_path)
    if if_mod_since and utils.cmp_mod_time(if_mod_since, last_mod):
        return sender(304, f'File not modified: {path}', path, {'Last-Modified': last_mod, 'Cache-Control': f'max-age={cache_age}'}, b'', 'info')
    
    # === Accept the request ===

    content = utils.get_content(full_path)
    content_type = CONTENT_TYPES[ext]
    content_len = os.path.getsize(full_path)

    resp_hdrs = {
        'Content-Type': content_type,
        'Content-Length': content_len,
        'Last-Modified': last_mod,
        'Cache-Control': f'max-age={cache_age}'
    }

    return sender(200, f'File sent: {path}', path, resp_hdrs, content, 'info')


def send(c_sock: socket.socket, resp: bytes) -> None:
    try:
        c_sock.sendall(resp)
    except Exception as e:
        log_err(f'[!] Error sending response: {e}')


def build(status: int, hdrs: dict, content: bytes, method: str, path: str) -> bytes:
    status_msg = {
        200: 'OK', 304: 'Not Modified', 400: 'Bad Request',
        403: 'Forbidden', 404: 'Not Found', 415: 'Unsupported Media Type'
    }.get(status, '')
    
    # Error page for non-200 and non-304 status
    if status != 200 and status != 304:
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error {status}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #D32F2F; }}
                .container {{ border: 1px solid #ccc; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Error {status}: {status_msg}</h1>
                <p>Path Access: {path}</p>
            </div>
        </body>
        </html>
        """
        content = error_html.encode()
        if 'Content-Type' not in hdrs:
            hdrs['Content-Type'] = 'text/html'
        hdrs['Content-Length'] = len(content)
    
    hdr_str = '\r\n'.join(f'{k}: {v}' for k, v in hdrs.items())
    resp = f'HTTP/1.1 {status} {status_msg}\r\n{hdr_str}\r\n\r\n'
    
    body = b''
    if method != 'HEAD':
        # only include body for 200
        # or 400, 403, 404, 415 (custom error page)
        if status == 200 or (status != 200 and status != 304):
            body = content
            
    return resp.encode() + body
