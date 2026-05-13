import os
from pathlib import Path
from urllib.request import urlretrieve
from dotenv import load_dotenv


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    url = os.getenv("CHECKPOINT_URL")
    if not url:
        raise RuntimeError("CHECKPOINT_URL is empty. Set it in .env before downloading.")
    out_dir = root / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "model_best.pth"
    urlretrieve(url, out_path)
    print(f"Checkpoint скачан в {out_path}")


if __name__ == "__main__":
    main()
