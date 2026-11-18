import pytest
import torch

from midline_agent.diagnostic.attention_mil import AttentionMIL


def test_attention_mil_combines_hidden_logits_and_modality_embeddings() -> None:
    model = AttentionMIL(
        logit_dimension=5,
        input_hidden_dimension=3,
        hidden_dimension=8,
        num_modalities=2,
        num_classes=4,
        dropout=0.0,
    ).eval()

    output = model(
        logits=torch.ones(2, 3, 5),
        hidden=torch.ones(2, 3, 3),
        modalities=torch.tensor([[0, 1, 0], [1, 0, 1]]),
        mask=torch.ones(2, 3, dtype=torch.bool),
    )

    assert output.logits.shape == (2, 4)
    assert output.evidence_weights.shape == (2, 3)
    assert output.attention_weights.shape == (2, 3)
    assert bool((output.evidence_weights > 0).all())
    torch.testing.assert_close(output.attention_weights.sum(dim=1), torch.ones(2))


def test_attention_mil_ignores_masked_padding() -> None:
    torch.manual_seed(7)
    model = AttentionMIL(hidden_dimension=8, num_modalities=2, dropout=0.0).eval()
    logits = torch.ones(1, 2, 5)
    modalities = torch.tensor([[0, -1]])
    mask = torch.tensor([[True, False]])
    changed_padding = logits.clone()
    changed_padding[:, 1] = 99

    first = model(logits=logits, modalities=modalities, mask=mask)
    second = model(logits=changed_padding, modalities=modalities, mask=mask)

    torch.testing.assert_close(first.logits, second.logits)
    assert first.evidence_weights[0, 0] > 0
    torch.testing.assert_close(first.evidence_weights[:, 1], torch.zeros(1))
    torch.testing.assert_close(first.attention_weights, torch.tensor([[1.0, 0.0]]))


def test_attention_mil_supports_hidden_features_without_logits() -> None:
    model = AttentionMIL(logit_dimension=0, input_hidden_dimension=6, hidden_dimension=8, dropout=0.0).eval()
    output = model(
        logits=None,
        hidden=torch.ones(1, 2, 6),
        modalities=torch.tensor([[0, 1]]),
        mask=torch.tensor([[True, True]]),
    )
    assert output.logits.shape == (1, 5)
    torch.testing.assert_close(output.evidence_weights, torch.ones(1, 2))
    torch.testing.assert_close(output.attention_weights.sum(dim=1), torch.ones(1))


def test_logit_evidence_calibration_is_distinct_from_slice_attention() -> None:
    model = AttentionMIL(hidden_dimension=8, dropout=0.0).eval()
    output = model(
        logits=torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0, 0.0]]]),
        modalities=torch.tensor([[0, 1]]),
        mask=torch.tensor([[True, False]]),
    )

    assert output.evidence_weights.data_ptr() != output.attention_weights.data_ptr()
    assert output.evidence_weights[0, 0] > 0
    assert output.evidence_weights[0, 1] == 0
    torch.testing.assert_close(output.attention_weights, torch.tensor([[1.0, 0.0]]))


def test_attention_mil_rejects_all_masked_bag() -> None:
    model = AttentionMIL(hidden_dimension=8)
    with pytest.raises(ValueError, match="valid instance"):
        model(
            logits=torch.ones(1, 2, 5),
            modalities=torch.tensor([[-1, -1]]),
            mask=torch.tensor([[False, False]]),
        )


def test_attention_mil_rejects_invalid_active_modality() -> None:
    model = AttentionMIL(hidden_dimension=8, num_modalities=2)
    with pytest.raises(ValueError, match="modality"):
        model(
            logits=torch.ones(1, 1, 5),
            modalities=torch.tensor([[2]]),
            mask=torch.tensor([[True]]),
        )


@pytest.mark.parametrize("feature_kind", ["logits", "hidden", "combined"])
def test_masked_extreme_features_are_zeroed_before_learned_layers(feature_kind: str) -> None:
    uses_logits = feature_kind in {"logits", "combined"}
    uses_hidden = feature_kind in {"hidden", "combined"}
    model = AttentionMIL(
        logit_dimension=5 if uses_logits else 0,
        input_hidden_dimension=3 if uses_hidden else 0,
        hidden_dimension=8,
        dropout=0.0,
    ).eval()
    maximum = torch.finfo(torch.float32).max
    logits = torch.tensor([[[1.0] * 5, [maximum] * 5]]) if uses_logits else None
    hidden = torch.tensor([[[1.0] * 3, [maximum] * 3]]) if uses_hidden else None
    modalities = torch.tensor([[0, -1]])
    mask = torch.tensor([[True, False]])
    observed: dict[str, torch.Tensor] = {}

    if model.evidence_calibration is not None:
        model.evidence_calibration.register_forward_pre_hook(
            lambda _module, inputs: observed.update(calibration=inputs[0].detach().clone())
        )
    model.feature_projection[0].register_forward_pre_hook(
        lambda _module, inputs: observed.update(projection=inputs[0].detach().clone())
    )

    output = model(logits=logits, hidden=hidden, modalities=modalities, mask=mask)

    if uses_logits:
        torch.testing.assert_close(observed["calibration"][:, 1], torch.zeros(1, 5))
    torch.testing.assert_close(observed["projection"][:, 1], torch.zeros_like(observed["projection"][:, 1]))
    assert bool(torch.isfinite(output.logits).all())
    assert bool(torch.isfinite(output.evidence_weights).all())
    assert bool(torch.isfinite(output.attention_weights).all())
