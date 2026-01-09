import os
from typing import Any
from colorama import Fore
from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt

# -----------------------------------
# |         CONSOLE LOGGING         |
# -----------------------------------

def print_dict(d: dict, indent: int = 4):
    def is_branch(x):
        return isinstance(x, (dict, list, tuple))

    def is_empty(x) -> bool:
        return (isinstance(x, dict) and not x) or (isinstance(x, (list, tuple)) and len(x) == 0)

    def brackets_for(x):
        if isinstance(x, dict):
            return "{", "}"
        return ("[", "]") if isinstance(x, list) else ("(", ")")

    def scalar_str(x: Any) -> str:
        try:
            if x.__class__.__module__.startswith("numpy") and hasattr(x, "item"):
                x = x.item()
        except Exception:
            pass
        if isinstance(x, str):
            return f'"{x}"'
        return str(x)

    def walk(obj: Any, level: int) -> None:
        pad = " " * (level * indent)
        if isinstance(obj, dict):
            if not obj:
                print(pad + f"{Fore.MAGENTA}{{}}{Fore.RESET}")
                return
            print(pad + f"{Fore.MAGENTA}{{{Fore.RESET}")
            for k, v in obj.items():
                line_pad = " " * ((level + 1) * indent)
                if is_branch(v):
                    if is_empty(v):
                        ob, cb = brackets_for(v)
                        print(f"{line_pad}{Fore.BLUE}{k}{Fore.RESET}: {Fore.MAGENTA}{ob}{cb}{Fore.RESET}")
                    else:
                        print(f"{line_pad}{Fore.BLUE}{k}{Fore.RESET}: {Fore.MAGENTA}→{Fore.RESET}")
                        walk(v, level + 1)
                else:
                    print(f"{line_pad}{Fore.BLUE}{k}{Fore.RESET}: {Fore.CYAN}{scalar_str(v)}{Fore.RESET}")
            print(pad + f"{Fore.MAGENTA}}}{Fore.RESET}")
        elif isinstance(obj, (list, tuple)):
            open_b, close_b = ("[", "]") if isinstance(obj, list) else ("(", ")")
            if len(obj) == 0:
                print(pad + f"{Fore.MAGENTA}{open_b}{close_b}{Fore.RESET}")
                return
            print(pad + f"{Fore.MAGENTA}{open_b}{Fore.RESET}")
            for i, v in enumerate(obj):
                line_pad = " " * ((level + 1) * indent)
                tag = f"{Fore.MAGENTA}[{i}]{Fore.RESET}"
                if is_branch(v):
                    if is_empty(v):
                        ob, cb = brackets_for(v)
                        print(f"{line_pad}{tag}: {Fore.MAGENTA}{ob}{cb}{Fore.RESET}")
                    else:
                        print(f"{line_pad}{tag}: {Fore.MAGENTA}→{Fore.RESET}")
                        walk(v, level + 1)
                else:
                    print(f"{line_pad}{tag}: {Fore.CYAN}{scalar_str(v)}{Fore.RESET}")
            print(pad + f"{Fore.MAGENTA}{close_b}{Fore.RESET}")
        else:
            print(pad + f"{Fore.CYAN}{scalar_str(obj)}{Fore.RESET}")

    walk(d, 0)


def print_warn(msg: str, header: str = "WARN", end: str = "\n"): 
    prefix = f"[{header}]: " if header else ""
    print(f"{Fore.YELLOW}{prefix}{msg}{Fore.RESET}", end=end)


def print_info(msg: str, header: str = "INFO", end: str = "\n"): 
    prefix = f"[{header}]: " if header else ""
    print(f"{Fore.BLUE}{prefix}{msg}{Fore.RESET}", end=end)


def print_good(msg: str, header: str = "OK", end: str = "\n"): 
    prefix = f"[{header}]: " if header else ""
    print(f"{Fore.GREEN}{prefix}{msg}{Fore.RESET}", end=end)


def print_error(msg: str, header: str = "ERR", end: str = "\n"): 
    prefix = f"[{header}]: " if header else ""
    print(f"{Fore.RED}{prefix}{msg}{Fore.RESET}", end=end)


# -----------------------------------
# |         PATH HANDLING           |
# -----------------------------------

def _get_project_root() -> Path:
    current: Path = Path(__file__).resolve()
    is_root = lambda p: (p/"pyproject.toml").exists() and (p/"src").exists()
    for parent in [current.parent, *current.parents]:
        if is_root(parent):
            print_good(f"Project Root @ {parent}")
            return parent

    print_error("Project root not found")
    raise RuntimeError("Project root not found")

project_root = _get_project_root()

def abs_path(path: str) -> Path:
    """
    Convert a path (assumed relative to project root) to an absolute path.
    Absolute inputs are returned unchanged. Environment variables and ~ are expanded.
    """
    input_path: Path = Path(os.path.expandvars(os.path.expanduser(path)))
    if input_path.is_absolute():
        return input_path
    return project_root / input_path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_txt(text: str, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_fig(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)