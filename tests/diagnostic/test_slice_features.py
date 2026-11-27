import pytest
import torch

from midline_agent.diagnostic.brainvlm_runtime import SliceFeatures


def test_slice_features_are_immutable_cpu_float32_snapshots() -> None:
    logits = torch.arange(5, dtype=torch.float16)
    hidden = torch.arange(4096, dtype=torch.float16)
    features = SliceFeatures(logits, hidden)
    logits.zero_()
    hidden.zero_()

    assert features.logits.dtype == torch.float32
    assert features.hidden.dtype == torch.float32
    assert features.logits.device.type == "cpu"
    assert features.logits.tolist() == list(map(float, range(5)))


@pytest.mark.parametrize(
    ("logits", "hidden"),
    [
        (torch.ones(4), torch.ones(4096)),
        (torch.ones(5), torch.ones(8)),
        (torch.full((5,), float("nan")), torch.ones(4096)),
    ],
)
def test_slice_features_reject_invalid_tensors(logits: torch.Tensor, hidden: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        SliceFeatures(logits, hidden)
