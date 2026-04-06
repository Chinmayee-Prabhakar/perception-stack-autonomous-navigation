"""Fetch ignored demo assets from stable upstream URLs; print hashes for provenance."""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen


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
    for path, (url, expected_digest) in ASSETS.items():
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


if __name__ == "__main__":
    main()
