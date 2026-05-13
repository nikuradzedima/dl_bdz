import argparse
from pathlib import Path
import torchaudio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument(
        "--splits", nargs="+", default=["train-clean-100", "test-clean"]
    )
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        print(f"Скачиваю LibriSpeech {split} в {root}")
        torchaudio.datasets.LIBRISPEECH(root=root, url=split, download=True)


if __name__ == "__main__":
    main()
