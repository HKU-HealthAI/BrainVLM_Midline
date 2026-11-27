import json
from pathlib import Path

from PIL import Image

from midline_agent.datasets.inference_annotations import load_inference_case


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = PROJECT_ROOT / "test_data" / "utsw_glioma_bt0323"


def test_public_case_contains_exactly_30_decodable_label_free_slices() -> None:
    annotation_path = CASE_ROOT / "annotations.json"
    records = json.loads(annotation_path.read_text(encoding="utf-8"))

    assert isinstance(records, list)
    assert len(records) == 30
    assert all(set(record) == {"patient_id", "image_path", "modality", "slice_index", "age"} for record in records)
    assert all(record["patient_id"] == "BT0323" and record["age"] == 38 for record in records)
    assert all(Path(record["image_path"]).as_posix().isascii() for record in records)
    assert not any(
        key in record
        for record in records
        for key in ("tumor_type", "who_class", "who_grade", "grade", "label", "logits", "probabilities", "prediction")
    )
    assert len({record["image_path"] for record in records}) == 30
    assert len(list((CASE_ROOT / "images").glob("*.png"))) == 30

    for record in records:
        image_path = CASE_ROOT / record["image_path"]
        assert image_path.is_file()
        with Image.open(image_path) as image:
            image.verify()


def test_public_case_loads_all_four_modalities() -> None:
    case = load_inference_case(CASE_ROOT / "annotations.json", CASE_ROOT)

    assert case.patient_id == "BT0323"
    assert case.age == 38.0
    assert {item.modality for item in case.slices} == {"T1", "T1CE", "T2", "FLAIR"}
    assert len(case.slices) == 30


def test_public_case_has_complete_tcia_attribution_and_derivative_notice() -> None:
    attribution = (CASE_ROOT / "ATTRIBUTION.md").read_text(encoding="utf-8")
    notices = (PROJECT_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    combined = attribution + notices

    for required in (
        "UTSW-Glioma",
        "Reddy",
        "The Cancer Imaging Archive",
        "https://doi.org/10.7937/DFAE-1B86",
        "https://www.cancerimagingarchive.net/collection/utsw-glioma/",
        "https://creativecommons.org/licenses/by/4.0/",
        "CC BY 4.0",
        "selected 2D PNG derivatives",
        "modified",
    ):
        assert required in combined


def test_public_case_does_not_bundle_generated_results() -> None:
    paths = [path.relative_to(CASE_ROOT).as_posix() for path in CASE_ROOT.rglob("*") if path.is_file()]

    assert all("result" not in path.lower() and "prediction" not in path.lower() for path in paths)
