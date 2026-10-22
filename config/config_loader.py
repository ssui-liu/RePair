"""YAML config loading."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent


def resolve_config_path(config_path: Optional[str] = None) -> Path:
    """Resolve a config given as an absolute path, or relative to the working directory,
    the repository root or `config/`. None selects the retrieval config.
    """
    if config_path is None:
        return CONFIG_DIR / "retrieval.yaml"
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        path = next((p for p in (PROJECT_ROOT / path, CONFIG_DIR / path) if p.exists()), path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return path


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    with resolve_config_path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
