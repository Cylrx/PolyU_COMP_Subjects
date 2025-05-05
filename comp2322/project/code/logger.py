from datetime import datetime
from argparse import Namespace

# ===== File Loggings =====

class FileLogger:
    def __init__(self, filename: str):
        self.filename = filename
        self.log_file = open(filename, 'a')

    def _log(self, message: str, end: str = '\n'):
        self.log_file.write(message + end)
        self.log_file.flush()
        
    def entry(
            self, 
            host: str, 
            port: int, 
            filename: str, 
            response_code: str,
        ) -> None: 
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._log(f'[{time}] {host}:{port} - {filename} - {response_code}')
    
    def close(self) -> None:
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._log(f'[{time}] Server Closed', end='\n\n')
        self.log_file.close()
    
    def start(self) -> None:
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._log(f'[{time}] Server Started')


# ===== Console Loggings =====
def print_args(args: Namespace):
    args: dict = vars(args)
    log_info('Server arguments:')

    for k, v in args.items():
        print(f"\t\033[94m{k}\033[0m: {v}") # blue + white
    print()

def log_info(message: str):
    print(f"\033[94m{message}\033[0m") # blue


def log_warn(message: str):
    print(f"\033[93m{message}\033[0m") # yellow


def log_err(message: str):
    print(f"\033[91m{message}\033[0m") # red


def log_succ(message: str):
    print(f"\033[92m{message}\033[0m") # green


def log_debug(message: str):
    escaped_message = message.encode("unicode_escape").decode("utf-8")
    print(f"\033[95m{escaped_message}\033[0m") # magenta


def main():
    log_info("This is an info message")
    log_warn("This is a warning message")
    log_err("This is an error message")
    log_succ("This is a success message")
    log_debug("\n\nThis is a debug\r\n message\t\0")


if __name__ == "__main__":
    main()

