import socket
import os
import datetime
import argparse
import concurrent.futures

import utils
from server import handle_cli, init_pool, shutdown_pool
from logger import log_info, log_err, log_succ, log_warn, print_args, FileLogger

def init_srv(host: str = '127.0.0.1', port: int = 80, max_req: int = 1000) -> socket.socket | None:
    srv_sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try: 
        if utils.check_bind(host, port): 
            srv_sock.bind((host, port))
            log_succ(f'[*] Server started on {host}:{port}')
            srv_sock.listen(max_req)
            return srv_sock


        for alt_port in range(8080, 65535):
            if utils.check_bind(host, alt_port):
                srv_sock.bind((host, alt_port))
                log_succ(f'Server started on {host}:{alt_port}')
                srv_sock.listen(max_req)
                return srv_sock

    except OSError as e:
        log_err(f'[!] Failed to start server - {e}')

    log_err('[!] Could not find any available ports')
    exit(1)


def main(args: argparse.Namespace):
    print_args(args)
    logger = FileLogger(args.log_file)

    srv_sock: socket.socket = init_srv(
        host=args.host,
        port=args.port,
        max_req=args.max_requests,
    )
    init_pool(args.max_threads)

    logger.start()

    try: 
        while True:
            c_sock: socket.socket
            c_addr: tuple[str, int]
            c_sock, c_addr = srv_sock.accept()
            handle_cli(c_sock, c_addr, logger, args.timeout, args.max_conn, args.cache_age)
    except OSError as e: log_err(f"[!] Server error - {e}")
    except KeyboardInterrupt: log_info("Shutting down server...")
    finally:
        shutdown_pool()
        logger.close()
        srv_sock.close()
        exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-threaded HTTP server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=80, help="Default port number")
    parser.add_argument("--max-requests", type=int, default=1000, help="Maximum number of requests handled in parallel (degree of threading)")
    parser.add_argument("--max-threads", type=int, default=100, help="Maximum number of threads")
    parser.add_argument("--timeout", type=int, default=15, help="Timeout for each connection")
    parser.add_argument("--max-conn", type=int, default=100, help="Maximum number of requests per connection")
    parser.add_argument("--cache-age", type=int, default=86400, help="Cache-control: max-age")
    parser.add_argument("--log-file", type=str, default="server.log", help="Log file name")
    args = parser.parse_args()

    main(args)
