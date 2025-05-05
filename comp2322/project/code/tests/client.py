import socket
import time
class HTTPClient:
    """
    Minimal small HTTP client over raw sockets for testing (See *Testing* section in the report)
    
    Usage:
        client = HTTPClient(...)
        s, h, b = client.request('GET', '/foo') # for a one‑shot request
        conn = client.open('HTTP/1.1') # for a persistent connection
        s1, h1, b1 = conn.send('GET', '/foo')
        s2, h2, b2 = conn.send('GET', '/bar')
        conn.close()
    """
    def __init__(self, host='127.0.0.1', port=8080, timeout=15):
        self.host = host
        self.port = port
        self.timeout = timeout

    def request(self, method, path, headers=None, body=None, version='HTTP/1.1', is_head=False):
        """
        Send a single HTTP request and close the connection.
        Returns: (status_code:int, headers:dict, body:bytes)
        """
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            req = self._build_request(method, path, headers, body, version)
            # body must be bytes if provided
            if body:
                sock.sendall(req.encode('utf-8') + body)
            else:
                sock.sendall(req.encode('utf-8'))
            return self._receive_response(sock, is_head)
    
    def request_N(self, method, path, headers=None, body=None, version='HTTP/1.1', is_head=False, N=100): 
        with socket.create_connection((self.host, self.port)) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256 * 1024 * 10)  # 256MB
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024 * 10)  # 256MB
            
            if headers is None:
                headers = {}
            headers['Connection'] = 'keep-alive'
            
            req = self._build_request(method, path, headers, body, version)
            for i in range(N): 
                if i == N - 1:
                    headers['Connection'] = 'close'
                    req = self._build_request(method, path, headers, body, version)
                sock.sendall(req.encode('utf-8'))
            return self.receive_all(sock)
        
    def receive_all(self, sock): 
        responses = []
        s = 200
        while s != None: 
            s, h, b = self._receive_response(sock, is_head=False)
            if s != None: 
                responses.append((s, h, b))
        return responses


    def open(self, version='HTTP/1.1'):
        """
        Open a persistent HTTP connection.  
        Returns an HTTPConnection that you can .send(...) multiple times.
        """
        return HTTPConnection(self.host, self.port, self.timeout, version)

    def _build_request(self, method, path, headers, body, version):
        lines = [f'{method} {path} {version}', f'Host: {self.host}']
        if body is not None:
            lines.append(f'Content-Length: {len(body)}')
        if headers:
            for k, v in headers.items():
                lines.append(f'{k}: {v}')
        lines.append('')
        lines.append('')
        return '\r\n'.join(lines)

    def _receive_response(self, sock, is_head=False):
        sock.settimeout(self.timeout)
        data = b''
        # read until end of headers
        while b'\r\n\r\n' not in data:
            chunk = sock.recv(10)
            if not chunk:
                break
            data += chunk
        
        if not data: 
            return None, {}, b''

        parts = data.split(b'\r\n\r\n', 1)
        header_blob = parts[0].decode('iso-8859-1').split('\r\n')
        rest = parts[1] if len(parts) > 1 else b''

        # parse status line
        status_line = header_blob[0]
        try:
            _, code_str, _ = status_line.split(' ', 2)
            status = int(code_str)
        except:
            status = None

        # parse headers
        hdrs = {}
        for line in header_blob[1:]:
            if ': ' in line:
                k, v = line.split(': ', 1)
                hdrs[k.lower()] = v

        # read body if Content-Length, and only when GET
        body = rest
        if not is_head and 'content-length' in hdrs:
            length = int(hdrs['content-length'])
            to_read = length - len(body)
            while to_read > 0:
                chunk = sock.recv(min(4096, to_read))
                if not chunk:
                    break
                body += chunk
                to_read -= len(chunk)
        return status, hdrs, body

class HTTPConnection:
    """
    A persistent HTTP connection.  
    Example:

        conn = HTTPClient(...).open('HTTP/1.1')
        s1, h1, b1 = conn.send('GET','/foo')
        s2, h2, b2 = conn.send('GET','/bar')
        conn.close()
    """
    def __init__(self, host, port, timeout, version):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.version = version
        self.sock = socket.create_connection((host, port), timeout=timeout)

    def send(self, method, path, headers=None, body=None):
        raw = self._build_request(method, path, headers, body)
        if body:
            self.sock.sendall(raw.encode('utf-8') + body)
        else:
            self.sock.sendall(raw.encode('utf-8'))
        return HTTPClient(self.host, self.port, self.timeout)._receive_response(self.sock)

    def _build_request(self, method, path, headers, body):
        lines = [f'{method} {path} {self.version}', f'Host: {self.host}']
        if body is not None:
            lines.append(f'Content-Length: {len(body)}')
        if headers:
            for k, v in headers.items():
                lines.append(f'{k}: {v}')
        lines.append('')
        lines.append('')
        return '\r\n'.join(lines)

    def close(self):
        try:
            self.sock.close()
        except:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
