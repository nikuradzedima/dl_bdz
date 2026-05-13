from pathlib import Path

ROOT_PATH = Path(__file__).absolute().resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT_PATH / path
