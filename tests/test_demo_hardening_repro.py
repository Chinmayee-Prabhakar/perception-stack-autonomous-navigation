"""Small offline regressions for portable D1 reproduction metadata and tools."""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest

from scripts.demo_hardening import check_assets, restore_inputs, verify_outputs


REPO = Path(__file__).resolve().parents[1]


def test_repo_relative_selection_discovery_from_another_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert restore_inputs.project_root() == REPO
    assert restore_inputs.selection_path(REPO) == REPO / "configs/demo_hardening/selection.json"
    assert restore_inputs.default_output(REPO) == REPO / "outputs/demo_hardening_reproduced/inputs"
    assert [record["frame_id"] for record in restore_inputs.load_selection(REPO)] == [
        "006263", "000201", "006781", "003783", "005481", "000185", "007200"]


def test_restore_verifies_zip_member_and_refuses_changed_existing_input(tmp_path):
    image = Image.new("RGB", (8, 4), (23, 45, 67))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    payload = stream.getvalue()
    record = {"frame_id": "000001", "source_member": "training/image_2/000001.png",
              "image_size": [8, 4], "image_bytes": len(payload),
              "image_sha256": hashlib.sha256(payload).hexdigest()}
    archive = tmp_path / "approved.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(record["source_member"], payload)
    target = tmp_path / "isolated" / "inputs"
    assert restore_inputs.restore(archive, target, [record])[0]["sha256"] == record["image_sha256"]
    assert (target / "000001.png").read_bytes() == payload
    (target / "000001.png").write_bytes(b"different")
    with pytest.raises(FileExistsError, match="Existing image differs"):
        restore_inputs.restore(archive, target, [record])


def test_calibration_derivation_rejects_changed_full_p2_translation():
    config = REPO / "configs/demo_hardening"
    intrinsics = json.loads((config / "006781_intrinsics.json").read_text(encoding="utf-8"))
    derivation = json.loads((config / "006781_derivation.json").read_text(encoding="utf-8"))
    selection = json.loads((config / "selection.json").read_text(encoding="utf-8"))
    frame = next(row for row in selection["frames"] if row["frame_id"] == "006781")
    check_assets.check_calibration(intrinsics, derivation, frame)
    altered = deepcopy(derivation)
    altered["p2_full"][0][3] += 1
    with pytest.raises(ValueError, match="camera-2 center derivation"):
        check_assets.check_calibration(intrinsics, altered, frame)


def test_historical_d0_is_optional_but_missing_requested_baseline_fails(tmp_path, monkeypatch):
    expected = verify_outputs.load_json(REPO / "configs/demo_hardening/expected_results.json")
    monkeypatch.setattr(verify_outputs, "snapshot", lambda output: expected)
    report = verify_outputs.verify(tmp_path, presentation_version="d1")
    assert report["presentation_version"] == "d1"
    assert report["historical_d0_comparison"].startswith("not_requested")
    with pytest.raises(FileNotFoundError, match="Historical D0 integrity check cannot run"):
        verify_outputs.verify(tmp_path, historical_d0=tmp_path / "missing", presentation_version="d1")


def test_versioned_presentation_preserves_scientific_and_historical_checks(tmp_path, monkeypatch):
    config = REPO / "configs/demo_hardening"
    accepted = verify_outputs.load_json(config / "expected_results.json")
    revised = verify_outputs.load_json(config / "presentation_r2.json")
    actual = deepcopy(accepted)
    for fid, overlay in revised["overlays"].items():
        actual["scenes"][fid]["overlay"] = overlay
    monkeypatch.setattr(verify_outputs, "snapshot", lambda output: actual)
    result = verify_outputs.verify(tmp_path)
    assert result["presentation_version"] == "r2"
    assert "scene, association and mask" in result["accepted_d1_scientific_manifest"]
    with pytest.raises(ValueError, match="Historical accepted D1 presentation"):
        verify_outputs.verify(tmp_path, presentation_version="d1")
    actual["scenes"]["000201"]["road_association_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="scientific scene, association or mask"):
        verify_outputs.verify(tmp_path)
