from midline_agent.diagnostic.who_grade import (
    parse_who_generation,
    vote_who_grade,
    vote_who_grade_for_tumor,
)


def test_who_parser_rejects_invalid_tumor_grade_pair() -> None:
    assert parse_who_generation("<tumor_5> <grade_4>").status == "failed"


def test_who_vote_breaks_tie_deterministically() -> None:
    result = vote_who_grade(["<tumor_4> <grade_2>", "<tumor_4> <grade_3>"])

    assert result.grade == "grade_2"
    assert result.vote_counts == {"grade_2": 1, "grade_3": 1}


def test_tumor_conditioned_vote_never_completes_invalid_tumor_grade_pair() -> None:
    result = vote_who_grade_for_tumor(["<tumor_5> <grade_4>"], "tumor_5")

    assert result.status == "failed"
    assert result.grade is None


def test_tumor_conditioned_vote_rejects_and_preserves_mismatched_valid_votes() -> None:
    result = vote_who_grade_for_tumor(
        [
            "<tumor_4> <grade_2>",
            "<tumor_2> <grade_3>",
            "<tumor_2> <grade_4>",
        ],
        "tumor_2",
    )

    assert result.status == "completed"
    assert result.tumor_type == "tumor_2"
    assert result.grade == "grade_3"
    assert result.vote_counts == {"grade_3": 1, "grade_4": 1}
    assert len(result.rejected_votes) == 1
    assert result.rejected_votes[0].tumor_type == "tumor_4"
    assert result.rejected_votes[0].grade == "grade_2"
