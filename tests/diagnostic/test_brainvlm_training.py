from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, DistributedSampler

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


class TinyModernPeftModel(TinyTrainModel):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Module()
        self.adapter.lora_A = nn.ModuleDict(
            {"default": nn.Linear(2, 1, bias=False)}
        )
        self.adapter.lora_B = nn.ModuleDict(
            {"default": nn.Linear(1, 2, bias=False)}
        )


class TinyAmbiguousPeftModel(TinyModernPeftModel):
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["adapter.lora_A.weight"] = state[
            "adapter.lora_A.default.weight"
        ]
        return state


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


def _legacy_manifest(model: nn.Module) -> dict[str, tuple[int, ...]]:
    return {
        name.replace(".lora_A.default.weight", ".lora_A.weight").replace(
            ".lora_B.default.weight", ".lora_B.weight"
        ): shape
        for name, shape in _manifest(model).items()
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
        "warmup_learning_rate": 1e-7,
        "weight_decay": 0.05,
        "warmup_steps": 2,
        "iterations_per_epoch": 4,
        "gradient_accumulation_steps": 2,
        "seed": 7,
        "num_workers": 0,
        "use_age": True,
        "upsample_ratio": 0.9,
        "amp": False,
        "device": "cuda:0",
    }
    value.update(updates)
    return value


def test_training_config_accepts_only_the_declared_schema(tmp_path: Path) -> None:
    config = BrainVlmTrainingConfig.from_mapping(_config(tmp_path))

    assert config.task == "tumor"
    assert config.epochs == 2
    assert config.annotations == (tmp_path / "annotations.json",)
    assert config.output_dir == tmp_path / "output"
    assert config.warmup_learning_rate == pytest.approx(1e-7)
    assert config.iterations_per_epoch == 4
    assert config.device == "cuda:0"
    assert config.distributed is False
    assert config.world_size is None
    assert config.dist_url is None
    with pytest.raises(TrainingContractError, match="unknown"):
        BrainVlmTrainingConfig.from_mapping(_config(tmp_path, mystery=True))


@pytest.mark.parametrize(
    "field",
    [
        "task",
        "annotations",
        "image_root",
        "llama_model",
        "biomedclip_pretrained",
        "output_dir",
        "epochs",
        "batch_size",
        "learning_rate",
        "min_learning_rate",
        "warmup_learning_rate",
        "weight_decay",
        "warmup_steps",
        "iterations_per_epoch",
        "gradient_accumulation_steps",
        "seed",
        "num_workers",
        "use_age",
        "upsample_ratio",
        "amp",
        "device",
    ],
)
def test_training_config_requires_every_required_field(
    tmp_path: Path,
    field: str,
) -> None:
    value = _config(tmp_path)
    value.pop(field)

    with pytest.raises(TrainingContractError, match="missing"):
        BrainVlmTrainingConfig.from_mapping(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", 1),
        ("annotations", "annotations.json"),
        ("image_root", 1),
        ("llama_model", 1),
        ("biomedclip_pretrained", 1),
        ("output_dir", 1),
        ("epochs", True),
        ("batch_size", 1.5),
        ("learning_rate", "0.001"),
        ("min_learning_rate", "0.0001"),
        ("warmup_learning_rate", "0.00001"),
        ("weight_decay", "0.05"),
        ("warmup_steps", 1.5),
        ("iterations_per_epoch", True),
        ("gradient_accumulation_steps", 1.5),
        ("seed", True),
        ("num_workers", 1.5),
        ("use_age", 1),
        ("upsample_ratio", "0.9"),
        ("amp", 1),
        ("device", 0),
    ],
)
def test_training_config_rejects_wrong_field_types(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    with pytest.raises(TrainingContractError, match=field.replace("_", " ")):
        BrainVlmTrainingConfig.from_mapping(_config(tmp_path, **{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task", "caption"),
        ("annotations", []),
        ("image_root", ""),
        ("llama_model", ""),
        ("biomedclip_pretrained", ""),
        ("output_dir", ""),
        ("epochs", 0),
        ("batch_size", 0),
        ("learning_rate", 0.0),
        ("min_learning_rate", -1.0),
        ("warmup_learning_rate", -1.0),
        ("weight_decay", -0.1),
        ("warmup_steps", -1),
        ("iterations_per_epoch", 0),
        ("gradient_accumulation_steps", 0),
        ("seed", -1),
        ("num_workers", -1),
        ("upsample_ratio", 1.1),
        ("device", "cuda"),
    ],
)
def test_training_config_rejects_out_of_range_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    with pytest.raises(TrainingContractError, match=field.replace("_", " ")):
        BrainVlmTrainingConfig.from_mapping(_config(tmp_path, **{field: value}))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"min_learning_rate": 2e-4}, "minimum learning rate"),
        ({"warmup_learning_rate": 2e-4}, "warmup learning rate"),
        ({"warmup_steps": 5}, "warmup steps"),
    ],
)
def test_training_config_rejects_inconsistent_schedule_values(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TrainingContractError, match=message):
        BrainVlmTrainingConfig.from_mapping(_config(tmp_path, **updates))


def test_training_config_accepts_strict_optional_fields(tmp_path: Path) -> None:
    config = BrainVlmTrainingConfig.from_mapping(
        _config(
            tmp_path,
            resume=str(tmp_path / "resume.pth"),
            distributed=True,
            world_size=2,
            dist_url="env://",
        )
    )

    assert config.resume == tmp_path / "resume.pth"
    assert config.initialization_checkpoint is None
    assert config.distributed is True
    assert config.world_size == 2
    assert config.dist_url == "env://"
    initialized = BrainVlmTrainingConfig.from_mapping(
        _config(
            tmp_path,
            initialization_checkpoint=str(tmp_path / "initial.pth"),
        )
    )
    assert initialized.initialization_checkpoint == tmp_path / "initial.pth"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resume", 1),
        ("initialization_checkpoint", 1),
        ("distributed", "yes"),
        ("world_size", True),
        ("dist_url", 1),
    ],
)
def test_training_config_rejects_wrong_optional_field_types(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    updates: dict[str, object] = {
        "distributed": True,
        "world_size": 2,
        "dist_url": "env://",
        field: value,
    }
    with pytest.raises(TrainingContractError, match=field.replace("_", " ")):
        BrainVlmTrainingConfig.from_mapping(_config(tmp_path, **updates))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"resume": ""}, "resume"),
        ({"initialization_checkpoint": ""}, "initialization checkpoint"),
        ({"distributed": True, "world_size": 0, "dist_url": "env://"}, "world size"),
        ({"distributed": True, "world_size": 2, "dist_url": "tcp://localhost"}, "dist url"),
        ({"distributed": True, "dist_url": "env://"}, "world size"),
        ({"distributed": True, "world_size": 2}, "dist url"),
        ({"world_size": 2}, "distributed"),
        ({"dist_url": "env://"}, "distributed"),
        (
            {
                "resume": "resume.pth",
                "initialization_checkpoint": "initial.pth",
            },
            "mutually exclusive",
        ),
    ],
)
def test_training_config_rejects_invalid_optional_field_combinations(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TrainingContractError, match=message):
        BrainVlmTrainingConfig.from_mapping(_config(tmp_path, **updates))


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
        warmup_learning_rate=0.2,
    )

    values = [scheduler.step() for _ in range(6)]

    assert values[0] == pytest.approx(0.2)
    assert values[1] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.1)
    assert values[2] > values[3] > values[4] > values[5]


def test_warmup_cosine_scheduler_resumes_at_the_next_global_step() -> None:
    full_optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(1))], lr=1.0)
    full = WarmupCosineScheduler(
        full_optimizer,
        warmup_steps=2,
        total_steps=6,
        minimum_learning_rate=0.1,
        warmup_learning_rate=0.2,
    )
    full_values = [full.step() for _ in range(6)]
    resumed_optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(1))], lr=1.0)
    resumed = WarmupCosineScheduler(
        resumed_optimizer,
        warmup_steps=2,
        total_steps=6,
        minimum_learning_rate=0.1,
        warmup_learning_rate=0.2,
    )

    resumed.restore(3)
    resumed_values = [resumed.step() for _ in range(3)]

    assert resumed_values == pytest.approx(full_values[3:])
    with pytest.raises(TrainingContractError, match="progress"):
        resumed.restore(7)


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


class ScalarLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, samples, reduction="mean"):
        return {"loss": self.weight * samples["value"]}


def _train_scalar_remainder(accumulation_steps: int) -> float:
    model = ScalarLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=0,
        total_steps=1,
        minimum_learning_rate=1.0,
        warmup_learning_rate=1.0,
    )
    train_epoch(
        model,
        [{"value": torch.tensor(1.0)}, {"value": torch.tensor(3.0)}],
        optimizer,
        scheduler,
        scaler=None,
        gradient_accumulation_steps=accumulation_steps,
        device=torch.device("cpu"),
        amp=False,
    )
    return float(model.weight.detach())


def test_train_epoch_normalizes_an_incomplete_accumulation_window() -> None:
    complete_window = _train_scalar_remainder(accumulation_steps=2)
    incomplete_window = _train_scalar_remainder(accumulation_steps=4)

    assert complete_window == pytest.approx(-1.0)
    assert incomplete_window == pytest.approx(complete_window)


def test_train_epoch_accumulates_and_steps_scaler_once() -> None:
    model = TinyTrainModel()
    configure_trainable_parameters(model)
    optimizer = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=0,
        total_steps=1,
        minimum_learning_rate=0.0,
        warmup_learning_rate=0.0,
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


def test_train_epoch_caps_iterations_before_accumulation() -> None:
    model = TinyTrainModel()
    configure_trainable_parameters(model)
    optimizer = build_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=0,
        total_steps=1,
        minimum_learning_rate=0.0,
        warmup_learning_rate=0.0,
    )

    result = train_epoch(
        model,
        [{"image": torch.ones(1, 1, 1, 2)} for _ in range(3)],
        optimizer,
        scheduler,
        scaler=None,
        gradient_accumulation_steps=2,
        max_iterations=2,
        device=torch.device("cpu"),
        amp=False,
    )

    assert result.optimizer_steps == 1


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
    restored_scheduler = WarmupCosineScheduler(
        restored_optimizer,
        warmup_steps=2,
        total_steps=10,
        minimum_learning_rate=1e-5,
        warmup_learning_rate=1e-6,
    )
    state = load_training_checkpoint(
        checkpoint,
        restored,
        restored_optimizer,
        scaler=None,
        scheduler=restored_scheduler,
        manifest=manifest,
    )
    assert state.start_epoch == 4
    assert state.global_step == 7
    assert restored_scheduler.completed_steps == 7
    assert load_brainvlm_checkpoint(checkpoint, restored, manifest=manifest) == tuple(sorted(manifest))


def test_training_checkpoint_normalizes_modern_peft_keys_for_inference(
    tmp_path: Path,
) -> None:
    model = TinyModernPeftModel()
    configure_trainable_parameters(model)
    manifest = _legacy_manifest(model)
    checkpoint = tmp_path / "checkpoint.pth"

    save_training_checkpoint(
        checkpoint,
        model,
        build_optimizer(model, learning_rate=1e-3, weight_decay=0.0),
        scaler=None,
        epoch=0,
        global_step=1,
        config={"task": "tumor"},
        manifest=manifest,
    )

    envelope = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(envelope["model"]) == set(manifest)
    assert all(".default.weight" not in name for name in envelope["model"])
    restored = TinyModernPeftModel()
    assert load_brainvlm_checkpoint(
        checkpoint,
        restored,
        manifest=manifest,
    ) == tuple(sorted(manifest))


def test_training_checkpoint_rejects_ambiguous_peft_state_keys(tmp_path: Path) -> None:
    model = TinyAmbiguousPeftModel()
    configure_trainable_parameters(model)

    with pytest.raises(TrainingContractError, match="collision"):
        save_training_checkpoint(
            tmp_path / "checkpoint.pth",
            model,
            build_optimizer(model, learning_rate=1e-3, weight_decay=0.0),
            scaler=None,
            epoch=0,
            global_step=1,
            config={"task": "tumor"},
            manifest=_legacy_manifest(model),
        )


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
            scheduler=None,
            manifest=manifest,
        )


def test_single_gpu_device_resolution_uses_configured_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.train_brainvlm_midline as command

    observed: dict[str, object] = {}
    config = BrainVlmTrainingConfig.from_mapping(_config(tmp_path, device="cuda:3"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda device: observed.update(device=device),
    )

    device = command._resolve_device(config, local_rank=0, world_size=1)

    assert device == torch.device("cuda:3")
    assert observed["device"] == torch.device("cuda:3")


def test_distributed_context_uses_validated_torchrun_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.train_brainvlm_midline as command

    config = BrainVlmTrainingConfig.from_mapping(
        _config(
            tmp_path,
            distributed=True,
            world_size=2,
            dist_url="env://",
        )
    )
    observed: dict[str, object] = {}
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(command, "_resolve_device", lambda *args, **kwargs: torch.device("cpu"))
    monkeypatch.setattr(command.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        command.dist,
        "init_process_group",
        lambda **kwargs: observed.update(kwargs),
    )

    context = command._distributed(config)

    assert (context.rank, context.world_size, context.local_rank) == (1, 2, 1)
    assert context.device == torch.device("cpu")
    assert context.owns_process_group is True
    assert observed == {
        "backend": "gloo",
        "init_method": "env://",
        "world_size": 2,
        "rank": 1,
    }


class TinyOrchestrationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": torch.tensor([float(index + 1), 1.0])}

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


@pytest.mark.parametrize(("rank", "writes_checkpoint"), [(0, True), (1, False)])
def test_public_run_training_orchestrates_seeded_distributed_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    writes_checkpoint: bool,
) -> None:
    import scripts.train_brainvlm_midline as command

    config = BrainVlmTrainingConfig.from_mapping(
        _config(
            tmp_path,
            epochs=2,
            iterations_per_epoch=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            distributed=True,
            world_size=2,
            dist_url="env://",
        )
    )
    dataset = TinyOrchestrationDataset()
    model = TinyTrainModel()
    configure_trainable_parameters(model)
    sampler_epochs: list[int] = []
    distributed_state = {"initialized": False, "barriers": 0, "destroyed": 0}

    class RecordingSampler(DistributedSampler):
        def set_epoch(self, epoch: int) -> None:
            sampler_epochs.append(epoch)
            super().set_epoch(epoch)

    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setattr(command, "_build_dataset", lambda value: dataset)
    monkeypatch.setattr(command, "_build_model", lambda value: model)
    monkeypatch.setattr(command, "CHECKPOINT_MANIFEST", _manifest(model))
    monkeypatch.setattr(command, "DistributedSampler", RecordingSampler)
    monkeypatch.setattr(
        command,
        "collate_task_samples",
        lambda samples: {"image": torch.stack([sample["image"] for sample in samples])},
    )
    monkeypatch.setattr(command, "_resolve_device", lambda *args, **kwargs: torch.device("cpu"))
    monkeypatch.setattr(
        command.nn.parallel,
        "DistributedDataParallel",
        lambda wrapped, **kwargs: wrapped,
    )
    monkeypatch.setattr(
        command.dist,
        "is_initialized",
        lambda: distributed_state["initialized"],
    )
    monkeypatch.setattr(
        command.dist,
        "init_process_group",
        lambda **kwargs: distributed_state.update(initialized=True),
    )
    monkeypatch.setattr(command.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(command.dist, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(command.dist, "all_reduce", lambda tensor, op=None: None)
    monkeypatch.setattr(
        command.dist,
        "barrier",
        lambda: distributed_state.update(
            barriers=distributed_state["barriers"] + 1
        ),
    )
    monkeypatch.setattr(
        command.dist,
        "destroy_process_group",
        lambda: distributed_state.update(
            initialized=False,
            destroyed=distributed_state["destroyed"] + 1,
        ),
    )

    result = command.run_training(config)

    assert dataset.epochs == [0, 1]
    assert sampler_epochs == [0, 1]
    assert result.rank == rank
    assert result.world_size == 2
    assert result.global_step == 2
    assert result.optimizer_steps == 1
    assert result.checkpoint_path.exists() is writes_checkpoint
    assert distributed_state == {"initialized": False, "barriers": 2, "destroyed": 1}


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
