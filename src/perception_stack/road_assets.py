"""Pinned official TwinLiteNetPlus Small BDD100K checkpoint."""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen


MODEL_ID = "twinlitenetplus-small-bdd100k"
SOURCE_REVISION = "90f1b8695ae311d5123b05f8534b2e11e42499d2"
CHECKPOINT_ID = "official-google-drive-small.pth"
CHECKPOINT_PATH = Path("models") / "twinlitenetplus_bdd100k_small.pth"
CHECKPOINT_URL = "https://drive.google.com/uc?export=download&id=1N7jNsa8P1dM-UevqPRsx6heUpN4L00FJ"
# Locally computed from the authors' file. The authors do not publish a SHA-256.
CHECKPOINT_SHA256 = "0dd3bd7356a5b70e1d21ae60cd09b55641b48bde738393177f94daff17ab6360"
CHECKPOINT_BYTES = 594777


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_road_checkpoint(path: str | Path = CHECKPOINT_PATH) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Road/lane checkpoint missing: {path}. Run `python scripts/download_assets.py --road`."
        )
    digest = file_sha256(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(
            f"Road/lane checkpoint SHA-256 mismatch for {path}: {digest}; "
            f"expected {CHECKPOINT_SHA256}. Remove the invalid file and rerun "
            "`python scripts/download_assets.py --road`."
        )
    return digest


def download_road_checkpoint(path: str | Path = CHECKPOINT_PATH) -> str:
    """Download the authors' exact Small file; verify before installing."""
    path = Path(path)
    if path.exists():
        return verify_road_checkpoint(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(CHECKPOINT_URL, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        digest = file_sha256(temporary)
        if digest != CHECKPOINT_SHA256:
            raise ValueError(
                f"Downloaded road/lane checkpoint SHA-256 mismatch: {digest}; "
                f"expected {CHECKPOINT_SHA256}. Source: {CHECKPOINT_URL}"
            )
        temporary.replace(path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)
