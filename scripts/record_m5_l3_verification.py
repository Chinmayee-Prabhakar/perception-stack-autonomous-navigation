"""Record actual M5-L3 test and preservation evidence from saved JUnit files."""

from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET

import numpy
import PIL
import torch


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/kitti_m5_l3/heldout"


def _suite(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError(f"No JUnit testsuite: {path}")
    return {key: int(suite.attrib.get(key, 0)) for key in ("tests", "errors", "failures", "skipped")}


def main() -> None:
    tests = {name: _suite(OUT / f"{name}_tests.xml") for name in ("focused", "offline", "integration")}
    expected = {"focused": (51, 0, 0, 0), "offline": (205, 0, 0, 3),
                "integration": (3, 0, 0, 0)}
    for name, counts in tests.items():
        if tuple(counts[key] for key in ("tests", "errors", "failures", "skipped")) != expected[name]:
            raise ValueError(f"Unexpected {name} JUnit counts: {counts}")
    preservation = json.loads((OUT / "preservation.json").read_text(encoding="utf-8"))
    if preservation["frozen_m5"]["compared"] != 309 or preservation["frozen_m5"]["mismatches"]:
        raise ValueError("Frozen M5 preservation incomplete")
    protocol_sha = sha256((REPO / "configs/kitti_lidar_protocol_v1.json").read_bytes()).hexdigest()
    if protocol_sha != "02451176def64af5ea83adc3c63df0ee37351176b008ec947f7da20c3f1f75fc":
        raise ValueError("Protocol digest changed")
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=REPO,
                                text=True, capture_output=True)
    if diff_check.returncode:
        raise ValueError(f"git diff --check failed: {diff_check.stdout} {diff_check.stderr}")
    record = {
        "test_results": tests,
        "test_commands": {
            "focused": ".venv\\Scripts\\python.exe -m pytest -q tests\\test_kitti_lidar.py tests\\test_kitti_lidar_real_data.py tests\\test_kitti_lidar_heldout_statistics.py -p no:cacheprovider --junitxml=outputs\\kitti_m5_l3\\heldout\\focused_tests.xml",
            "offline": ".venv\\Scripts\\python.exe -m pytest -q -p no:cacheprovider --junitxml=outputs\\kitti_m5_l3\\heldout\\offline_tests.xml",
            "integration": "PERCEPTION_RUN_INTEGRATION=1 .venv\\Scripts\\python.exe -m pytest -q tests\\test_depth_integration.py tests\\test_road_integration.py tests\\test_fusion_integration.py -p no:cacheprovider --junitxml=outputs\\kitti_m5_l3\\heldout\\integration_tests.xml",
            "preservation": ".venv\\Scripts\\python.exe scripts\\verify_m5_l3_preservation.py",
            "git_diff_check": "git diff --check",
        },
        "frozen_m5_files_compared": preservation["frozen_m5"]["compared"],
        "accepted_m5_l2_pilot_files_compared": preservation["accepted_m5_l2_pilot"]["compared"],
        "accepted_m5_l2_preflight_files_compared": preservation["accepted_m5_l2_preflight"]["compared"],
        "protocol_v1_sha256": protocol_sha,
        "git_diff_check": "passed",
        "versions": {"python": platform.python_version(), "numpy": numpy.__version__,
                     "pillow": PIL.__version__, "torch": torch.__version__},
    }
    (OUT / "verification.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"tests": tests, "protocol_v1_sha256": protocol_sha}, indent=2))


if __name__ == "__main__":
    main()
