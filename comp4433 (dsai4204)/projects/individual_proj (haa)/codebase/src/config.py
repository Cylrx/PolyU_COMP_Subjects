import os
import yaml
from typing import Any, Iterable, Optional
from copy import deepcopy

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            object.__setattr__(cls._instance, '_base', {})
            object.__setattr__(cls._instance, '_over', {})
        return cls._instance
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        inst = cls()
        with open(path, 'r') as f:
            base = yaml.safe_load(f) or {}
        object.__setattr__(inst, '_base', base)
        object.__setattr__(inst, '_over', {})
        object.__setattr__(inst, '_path', path)
        return inst
    
    def update(self, **kwargs):
        for k in kwargs:
            if k not in self._base:
                raise KeyError(k)
        self._over.update(kwargs)
    
    def __getattr__(self, name: str) -> Any:
        if name in self._over:
            return self._over[name]
        if name in self._base:
            return self._base[name]
        raise AttributeError(f"No such config key: {name}. Check {self._path} for available keys.")
    
    def __setattr__(self, name: str, value: Any):
        if name in ('_base', '_over'):
            object.__setattr__(self, name, value)
        else:
            if name not in self._base:
                raise AttributeError(f"Unknown config key '{name}'")
            self._over[name] = value

    # experiments: deep-merge baseline with overlays (later overlays win)
    def exp(self, overlays: Optional[Iterable[str]] = None) -> dict:
        exps = self._base["experiments"]
        base = deepcopy(exps["baseline"])
        if overlays is None: return base
        if isinstance(overlays, (str, bytes)): overlays = [overlays]
        for name in overlays:
            assert name in exps["overlays"], f"{name} not found"
            o = exps["overlays"][name]
            _dmerge(base, o)
        assert "init" in base and "fit" in base
        return base


def _dmerge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _dmerge(a[k], v)
        else:
            a[k] = deepcopy(v)
    return a


cwd = os.path.dirname(__file__)
cfg_path = os.path.join(cwd, "config.yaml")
cfg = Config.from_yaml(cfg_path)