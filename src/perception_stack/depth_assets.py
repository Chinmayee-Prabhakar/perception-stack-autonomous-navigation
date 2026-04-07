"""Pinned assets for the official outdoor metric Depth Anything V2 Small model."""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen


MODEL_ID = "depth-anything-v2-metric-vkitti-vits"
CHECKPOINT_NAME = "depth_anything_v2_metric_vkitti_vits.pth"
CHECKPOINT_PATH = Path("models") / CHECKPOINT_NAME
CHECKPOINT_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/"
    f"resolve/main/{CHECKPOINT_NAME}"
)
# Published on the official Hugging Face file page, independently of our download.
CHECKPOINT_SHA256 = "9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb"
SOURCE_REVISION = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: str | Path = CHECKPOINT_PATH) -> str:
    """Return the hash only for the exact published metric VKITTI Small file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Metric checkpoint missing: {path}. Run `python scripts/download_assets.py --depth`."
        )
    digest = file_sha256(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(
            f"Metric checkpoint SHA-256 mismatch for {path}: {digest}; "
            f"expected {CHECKPOINT_SHA256}. Remove the invalid file and rerun "
            "`python scripts/download_assets.py --depth`."
        )
    return digest


def download_depth_checkpoint(path: str | Path = CHECKPOINT_PATH) -> str:
    """Download only if absent; verify before atomically installing the file."""
    path = Path(path)
    if path.exists():
        return verify_checkpoint(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(CHECKPOINT_URL, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        digest = file_sha256(temporary)
        if digest != CHECKPOINT_SHA256:
            raise ValueError(
                f"Downloaded metric checkpoint SHA-256 mismatch: {digest}; "
                f"expected {CHECKPOINT_SHA256}. Source: {CHECKPOINT_URL}"
            )
        temporary.replace(path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)
