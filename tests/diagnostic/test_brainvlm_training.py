from pathlib import Path

import pytest
import torch
from torch import nn

from midline_agent.diagnostic.brainvlm_midline.training import (
    BrainVlmTrainingConfig,
    TrainingContractError,
    WarmupCosineScheduler,
    build_optimizer,
    configure_trainable_parameters,
    load_training_checkpoint,
    save_training_checkpoint,
    train_epoch,
)
from midline_agent.diagnostic.checkpoints import load_brainvlm_checkpoint


class TinyTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.foundation = nn.Linear(2, 2)
        self.visual_encoder = nn.Module()
        self.visual_encoder.norm = nn.LayerNorm(2)
        self.llama_proj_mlps = nn.Sequential(nn.Linear(2, 2), nn.GELU(), nn.Linear(2, 2))
        self.v_q_project = nn.Linear(2, 2)
        self.cross_attention = nn.Linear(2, 2)
        self.adapter = nn.Module()
        self.adapter.lora_A = nn.Linear(2, 1, bias=False)
        self.adapter.lora_B = nn.Linear(1, 2, bias=False)

    def forward(self, samples, reduction="mean"):
        value = samples["image"].reshape(samples["image"].shape[0], -1)[:, :2]
        output = self.llama_proj_mlps(value)
        loss = output.square().mean(dim=1)
        return {"loss": loss.mean() if reduction == "mean" else loss}


def _manifest(model: nn.Module) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(value.shape)
        for name, value in model.state_dict().items()
        if any(
            marker in name
            for marker in (
                "lora_A",
                "lora_B",
                "visual_encoder.norm",
                "llama_proj_mlps",
                "v_q_project",
                "cross_attention",
            )
        )
    }


def _config(tmp_path: Path, **updates) -> dict[str, object]:
    value: dict[str, object] = {
        "task": "tumor",
        "annotations": [str(tmp_path / "annotations.json")],
        "image_root": str(tmp_path / "images"),
        "llama_model": str(tmp_path / "llama"),
        "biomedclip_pretrained": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "output_dir": str(tmp_path / "output"),
        "epochs": 2,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "min_learning_rate": 1e-6,
        "weight_decay": 0.05,
        "warmup_steps": 2,
        "gradient_accumulation_steps": 2,
        "seed": 7,
        "num_workers": 0,
        "use_age": True,
        "upsample_ratio": 0.9,
        "amp": False,
    }
    value.update(updates)
    return value


def test_training_config_accepts_only_the_declared_schema(tmp_path: Path) -> None:
    config = BrainVlmTrainingConfig.from_mapping(_config(tmp_path))

    assert config.task == "tumor"
    assert config.epochs == 2
    assert config.annotations == (tmp_path / "annotations.json",)
    assert config.output_dir == tmp_path / "output"
    with pytest.raises(TrainingContractError, match="unknown"):
        BrainVlmTrainingConfig.from_mapping(_config(tmp_path, mystery=True))
    invalid = _config(tmp_path)
    invalid.pop("task")
    with pytest.raises(TrainingContractError, match="missing"):
        BrainVlmTrainingConfig.from_mapping(invalid)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", "caption"),
        ("epochs", 0),
        ("batch_size", True),
        ("learning_rate", 0.0),
        ("min_learning_rate", -1.0),
        ("weight_decay", -0.1),
        ("warmup_steps", -1),
        ("gradient_accumulation_steps", 0),
        ("num_workers", -1),
        ("upsample_ratio", 1.1),
        ("amp", "yes"),
    ],
)
def test_training_config_rejects_invalid_values(tmp_path: Path, field: str, value: object) -> None:
    with pytest.raises(TrainingContractError, match=field.replace("_", " ")):
        BrainVlmTrainingConfig.from_mapping(_config(tmp_path, **{field: value}))


def test_parameter_policy_and_optimizer_groups_are_exact() -> None:
    model = TinyTrainModel()
    names = configure_trainable_parameters(model)
    optimizer = build_optimizer(model, learning_rate=1e-3, weight_decay=0.1)

    assert names == tuple(sorted(_manifest(model)))
    assert not model.foundation.weight.requires_grad
    assert model.visual_encoder.norm.weight.requires_grad
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert {group["weight_decay"] for group in optimizer.param_groups} == {0.0, 0.1}


def test_warmup_cosine_scheduler_reaches_base_then_minimum() -> None:
    parameter = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=2,
        total_steps=6,
        minimum_learning_rate=0.1,
    )

    values = [scheduler.step() for _ in range(6)]

    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.1)
    assert values[2] > values[3] > values[4] > values[5]


class RecordingScaler:
    def __init__(self) -> None:
        self.steps = 0
        self.updates = 0

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        self.steps += 1
        optimizer.step()

    def update(self):
        self.updates += 1


def test_train_epoch_accumulates_and_steps_scaler_once() -> None:
    model = TinyTrainModel()
    configure_trainable_parameters(model)
    optimizer = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=0,
        total_steps=1,
        minimum_learning_rate=0.0,
    )
    scaler = RecordingScaler()
    batches = [
        {"image": torch.ones(1, 1, 1, 2)},
        {"image": torch.full((1, 1, 1, 2), 2.0)},
    ]

    result = train_epoch(
        model,
        batches,
        optimizer,
        scheduler,
        scaler=scaler,
        gradient_accumulation_steps=2,
        device=torch.device("cpu"),
        amp=False,
    )

    assert result.optimizer_steps == 1
    assert scaler.steps == scaler.updates == 1
    assert result.mean_loss >= 0


def test_training_checkpoint_is_trainable_only_and_resumable(tmp_path: Path) -> None:
    model = TinyTrainModel()
    configure_trainable_parameters(model)
    manifest = _manifest(model)
    optimizer = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    checkpoint = tmp_path / "checkpoint.pth"

    save_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        scaler=None,
        epoch=3,
        global_step=7,
        config={"task": "tumor"},
        manifest=manifest,
    )
    envelope = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert set(envelope["model"]) == set(manifest)
    assert "foundation.weight" not in envelope["model"]
    restored = TinyTrainModel()
    configure_trainable_parameters(restored)
    restored_optimizer = build_optimizer(restored, learning_rate=1e-3, weight_decay=0.0)
    state = load_training_checkpoint(
        checkpoint,
        restored,
        restored_optimizer,
        scaler=None,
        manifest=manifest,
    )
    assert state.start_epoch == 4
    assert state.global_step == 7
    assert load_brainvlm_checkpoint(checkpoint, restored, manifest=manifest) == tuple(sorted(manifest))


def test_resume_rejects_inference_only_checkpoint(tmp_path: Path) -> None:
    model = TinyTrainModel()
    configure_trainable_parameters(model)
    manifest = _manifest(model)
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"model": {name: model.state_dict()[name] for name in manifest}}, checkpoint)

    with pytest.raises(TrainingContractError, match="resume"):
        load_training_checkpoint(
            checkpoint,
            model,
            build_optimizer(model, learning_rate=1e-3, weight_decay=0.0),
            scaler=None,
            manifest=manifest,
        )


def test_cli_rejects_task_mismatch_before_constructing_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.train_brainvlm_midline as command

    config = tmp_path / "config.yaml"
    import yaml

    config.write_text(yaml.safe_dump(_config(tmp_path, task="tumor")), encoding="ascii")
    monkeypatch.setattr(
        "sys.argv",
        ["train_brainvlm_midline.py", "--task", "who", "--config", str(config)],
    )

    with pytest.raises(TrainingContractError, match="task"):
        command.main()
