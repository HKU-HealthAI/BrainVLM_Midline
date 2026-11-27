import pytest

from midline_agent.diagnostic.brainvlm_midline.prompts import (
    ALLOWED_GRADES,
    TUMOR_LABELS,
    TUMOR_TRAINING_INSTRUCTIONS,
    WHO_TRAINING_INSTRUCTIONS,
    age_context,
    build_tumor_inference_prompt,
    build_tumor_training_prompt,
    build_who_training_prompt,
    validate_tumor_grade,
)


def test_tumor_inference_prompt_preserves_checkpoint_suffix() -> None:
    prompt = build_tumor_inference_prompt(age=None)

    assert prompt.endswith("ASSISTANT:<tumor_")
    assert prompt.count("<ImageHere>") == 1
    assert "'IDH-mutated astrocytoma': '<tumor_4>'" in prompt


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (None, ""),
        (0, "This is a child patient. "),
        (18, "This is a child patient. "),
        (18.01, "This is a adult patient. "),
        (90, "This is a adult patient. "),
    ],
)
def test_age_context_uses_training_boundary(age: float | None, expected: str) -> None:
    assert age_context(age) == expected


@pytest.mark.parametrize("age", [float("nan"), float("inf"), -1, True, "18"])
def test_age_context_rejects_invalid_values(age: object) -> None:
    with pytest.raises(ValueError, match="age"):
        age_context(age)  # type: ignore[arg-type]


def test_instruction_pools_match_released_training_tasks() -> None:
    assert TUMOR_TRAINING_INSTRUCTIONS == (
        "Please predict the WHO classification for this patient based on the MRI image.",
        "Based on the MRI image, predict the WHO classification.",
        "What is the WHO classification for this patient?",
    )
    assert WHO_TRAINING_INSTRUCTIONS == (
        "Please classify both the WHO tumor type and WHO grade for this midline glioma patient based on the MRI image.",
        "Based on the MRI image, determine both the tumor classification and grade for this case.",
        "Analyze this midline glioma case and provide both the WHO tumor type and WHO grade classification.",
        "From the MRI scan, classify both the tumor type (tumor_1 to tumor_5) and the corresponding WHO grade.",
    )


def test_training_prompts_insert_age_and_instruction_without_changing_context() -> None:
    tumor = build_tumor_training_prompt(TUMOR_TRAINING_INSTRUCTIONS[1], age=12)
    who = build_who_training_prompt(WHO_TRAINING_INSTRUCTIONS[2], age=45)

    assert "USER:This is a child patient. Based on the MRI image" in tumor
    assert tumor.endswith("ASSISTANT:")
    assert "USER:This is a adult patient. Analyze this midline glioma case" in who
    assert who.endswith("ASSISTANT:")
    assert "Your output format: '<tumor_X> <grade_Y>'" in who


def test_training_prompt_rejects_an_instruction_from_the_other_task() -> None:
    with pytest.raises(ValueError, match="tumor instruction"):
        build_tumor_training_prompt(WHO_TRAINING_INSTRUCTIONS[0])
    with pytest.raises(ValueError, match="WHO instruction"):
        build_who_training_prompt(TUMOR_TRAINING_INSTRUCTIONS[0])


def test_tumor_and_grade_vocabularies_are_canonical() -> None:
    assert TUMOR_LABELS == (
        "<tumor_1>",
        "<tumor_2>",
        "<tumor_3>",
        "<tumor_4>",
        "<tumor_5>",
    )
    assert ALLOWED_GRADES == {
        "<tumor_1>": ("<grade_4>",),
        "<tumor_2>": ("<grade_2>", "<grade_3>", "<grade_4>"),
        "<tumor_3>": ("<grade_4>",),
        "<tumor_4>": ("<grade_2>", "<grade_3>", "<grade_4>"),
        "<tumor_5>": ("<grade_2>", "<grade_3>"),
    }


def test_validate_tumor_grade_rejects_unknown_and_impossible_pairs() -> None:
    validate_tumor_grade("<tumor_4>", "<grade_3>")

    with pytest.raises(ValueError, match="tumor label"):
        validate_tumor_grade("tumor_4", "<grade_3>")
    with pytest.raises(ValueError, match="grade label"):
        validate_tumor_grade("<tumor_4>", "grade_3")
    with pytest.raises(ValueError, match="not valid"):
        validate_tumor_grade("<tumor_5>", "<grade_4>")
