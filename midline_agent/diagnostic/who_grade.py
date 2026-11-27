"""Parse and aggregate constrained WHO tumor/grade generations."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


ALLOWED_GRADES = {
    "tumor_1": {"grade_4"},
    "tumor_2": {"grade_2", "grade_3", "grade_4"},
    "tumor_3": {"grade_4"},
    "tumor_4": {"grade_2", "grade_3", "grade_4"},
    "tumor_5": {"grade_2", "grade_3"},
}
TOKEN_PATTERN = re.compile(r"<(tumor_[1-5])>[ \t]*<(grade_[234])>")


@dataclass(frozen=True)
class WhoGradeResult:
    status: str
    tumor_type: str | None = None
    grade: str | None = None
    reason: str = ""
    vote_counts: dict[str, int] | None = None
    rejected_votes: tuple["RejectedWhoVote", ...] = ()
    invalid_vote_count: int = 0


@dataclass(frozen=True)
class RejectedWhoVote:
    slice_index: int
    tumor_type: str
    grade: str
    reason: str = "tumor_mismatch"


def parse_who_generation(raw_text: str) -> WhoGradeResult:
    match = TOKEN_PATTERN.fullmatch(raw_text.strip())
    if not match:
        return WhoGradeResult(status="failed", reason="missing constrained tumor/grade tokens")
    tumor_type, grade = match.groups()
    if grade not in ALLOWED_GRADES[tumor_type]:
        return WhoGradeResult(status="failed", tumor_type=tumor_type, grade=grade, reason="invalid tumor-grade pair")
    return WhoGradeResult(status="completed", tumor_type=tumor_type, grade=grade)


def vote_who_grade(raw_predictions: list[str]) -> WhoGradeResult:
    parsed = [parse_who_generation(value) for value in raw_predictions]
    valid = [result for result in parsed if result.status == "completed" and result.grade is not None]
    invalid_vote_count = len(parsed) - len(valid)
    if not valid:
        return WhoGradeResult(
            status="failed",
            reason="no valid WHO-grade predictions",
            vote_counts={},
            invalid_vote_count=invalid_vote_count,
        )
    counts = Counter(result.grade for result in valid)
    grade = min((name for name, count in counts.items() if count == max(counts.values())), key=lambda name: int(name[-1]))
    tumor_type = min(result.tumor_type for result in valid if result.grade == grade)
    return WhoGradeResult(
        status="completed",
        tumor_type=tumor_type,
        grade=grade,
        vote_counts=dict(sorted(counts.items())),
        invalid_vote_count=invalid_vote_count,
    )


def vote_who_grade_for_tumor(raw_predictions: list[str], tumor_type: str) -> WhoGradeResult:
    if tumor_type not in ALLOWED_GRADES:
        raise ValueError("unsupported tumor type for WHO-grade voting")
    parsed = [parse_who_generation(value) for value in raw_predictions]
    invalid_vote_count = sum(result.status != "completed" for result in parsed)
    matching = [
        result
        for result in parsed
        if result.status == "completed" and result.tumor_type == tumor_type and result.grade is not None
    ]
    rejected = tuple(
        RejectedWhoVote(index, result.tumor_type, result.grade)
        for index, result in enumerate(parsed)
        if result.status == "completed"
        and result.tumor_type is not None
        and result.tumor_type != tumor_type
        and result.grade is not None
    )
    if not matching:
        return WhoGradeResult(
            status="failed",
            tumor_type=tumor_type,
            reason="no valid WHO-grade predictions for final tumor class",
            vote_counts={},
            rejected_votes=rejected,
            invalid_vote_count=invalid_vote_count,
        )
    counts = Counter(result.grade for result in matching)
    grade = min(
        (name for name, count in counts.items() if count == max(counts.values())),
        key=lambda name: int(name[-1]),
    )
    return WhoGradeResult(
        status="completed",
        tumor_type=tumor_type,
        grade=grade,
        vote_counts=dict(sorted(counts.items())),
        rejected_votes=rejected,
        invalid_vote_count=invalid_vote_count,
    )
