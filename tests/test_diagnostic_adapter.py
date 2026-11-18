import numpy as np

from midline_agent.datasets.prompts import PROMPT_VERSION
from midline_agent.diagnostic.adapters import WhoGradeRequest, WhoGradeSliceResult


def test_who_grade_contract_is_versioned_and_carries_slice_logits() -> None:
    request = WhoGradeRequest(slice_id="s-1", image=np.zeros((3, 4, 4)))
    result = WhoGradeSliceResult(
        raw_text="<tumor_1> <grade_4>",
        logits=np.array([5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        prompt_version=request.prompt_version,
    )

    assert request.prompt_version == PROMPT_VERSION
    assert result.raw_text == "<tumor_1> <grade_4>"
    assert result.logits.shape == (5,)
    assert result.prompt_version == PROMPT_VERSION
