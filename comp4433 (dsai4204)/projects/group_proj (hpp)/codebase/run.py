from pathlib import Path
from urllib.request import urlretrieve
from importlib.metadata import version, PackageNotFoundError
from src.hpp.main import load_hpp_data, main
from src.hpp.utils import ensure_dir, print_info, print_good, print_error, print_warn

def check_tabpfn_version() -> bool:
    try:
        v = version('tabpfn')
        if v != '2.1.0':
            print_warn(f"You are using an incompatible version of TabPFN ({v}). It will likely yield SUBOPTIMAL results!")
            print_warn(f"Please pin to tabpfn==2.1.0, or use the existing poetry.lock file.")
            print_warn(f"See https://github.com/PriorLabs/TabPFN/issues/458 for more details.")
            if input(f"Are you sure you want to continue anyway? (y/n): ").lower().strip() == "y":
                return False
            exit(1)
        else: 
            print_good(f"TabPFN version: {v}")
            return True
    except PackageNotFoundError:
        print_error(f"TabPFN not found. Please install tabpfn==2.1.0, or use the existing poetry.lock file via `poetry install`.\nSee https://github.com/PriorLabs/TabPFN/issues/458 for more details.")
        return False

def preliminary(): 
    # This is necessary for TabPFNv2 to work in extreme preset
    # Alternatively pin to tabpfn==2.1.0 which works fine
    # See: https://github.com/PriorLabs/TabPFN/issues/458
    # Download model from https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor-5wof9ojf.ckpt 
    # store at ~/.cache/tabpfn/tabpfn-v2-regressor-5wof9ojf.ckpt

    model_url = "https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor-5wof9ojf.ckpt"
    cache_dir = Path.home() / ".cache" / "tabpfn"
    model_path = cache_dir / "tabpfn-v2-regressor-5wof9ojf.ckpt"
    
    if model_path.exists():
        print_good(f"Model already exists at {model_path}")
        return
    
    ensure_dir(cache_dir)
    print_info(f"Downloading missing model:\t{model_url}...")
    print_info(f"Target location:\t{model_path}")
    try:
        urlretrieve(model_url, model_path)
        print_good(f"Model successfully downloaded to {model_path}")
    except Exception as e:
        print_error(f"Failed to download model: {e}")
        raise


if __name__ == "__main__":
    if not check_tabpfn_version():
        preliminary()
    train, test = load_hpp_data()
    main(train, test)

