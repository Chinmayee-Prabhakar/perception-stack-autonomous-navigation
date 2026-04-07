"""Fetch ignored demo assets from stable upstream URLs; print hashes for provenance."""

import argparse
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

from perception_stack.depth_assets import CHECKPOINT_PATH, CHECKPOINT_URL, download_depth_checkpoint


ASSETS = {
    Path("models/yolo11n-seg.pt"): (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
        "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    ),
    Path("data/examples/bus.jpg"): (
        "https://ultralytics.com/images/bus.jpg",
        "c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download verified, ignored demo/model assets")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--depth", action="store_true", help="download only the metric VKITTI Small checkpoint")
    selection.add_argument("--all", action="store_true", help="download the existing assets and metric checkpoint")
    args = parser.parse_args()
    for path, (url, expected_digest) in (() if args.depth else ASSETS.items()):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise ValueError(f"Unexpected SHA-256 for {path}: {digest}")
        print(f"{path}: {path.stat().st_size} bytes; sha256={digest}; source={url}")
    if args.depth or args.all:
        digest = download_depth_checkpoint()
        print(f"{CHECKPOINT_PATH}: {CHECKPOINT_PATH.stat().st_size} bytes; "
              f"sha256={digest}; source={CHECKPOINT_URL}")


if __name__ == "__main__":
    main()
