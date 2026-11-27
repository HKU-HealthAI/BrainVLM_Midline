import ast
import hashlib
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _toml_list(name: str) -> list[str]:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="ascii")
    match = re.search(rf"^{re.escape(name)}\s*=\s*(\[.*?\])", pyproject, re.MULTILINE | re.DOTALL)
    assert match is not None, f"missing pyproject list: {name}"
    value = ast.literal_eval(match.group(1))
    assert isinstance(value, list)
    return value


def test_diagnostic_and_all_extras_publish_compatible_runtime_constraints() -> None:
    expected = {
        "torch>=2.1,<2.2",
        "torchvision>=0.16,<0.17",
        "transformers>=4.46,<4.47",
        "peft>=0.2,<0.3",
        "open-clip-torch>=2.24,<2.25",
        "omegaconf>=2.3,<2.4",
        "einops>=0.8,<0.9",
        "Pillow>=10,<12",
        "numpy>=1.24,<2",
        "safetensors>=0.4,<1",
        "PyYAML>=6,<7",
    }

    assert set(_toml_list("diagnostic")) == expected
    assert expected < set(_toml_list("all"))


def test_reference_requirements_pin_the_captured_python_39_environment() -> None:
    path = PROJECT_ROOT / "requirements" / "diagnostic-reference.txt"
    lines = {
        line.strip()
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert lines == {
        "torch==2.1.2",
        "torchvision==0.16.2",
        "transformers==4.46.3",
        "peft==0.2.0",
        "open-clip-torch==2.24.0",
        "omegaconf==2.3.0",
        "einops==0.8.1",
        "Pillow==11.3.0",
        "numpy==1.26.4",
        "safetensors==0.6.2",
        "PyYAML==6.0",
        "tokenizers==0.20.3",
        "accelerate==0.25.0",
    }
    content = path.read_text(encoding="ascii")
    assert "platform-specific" in content
    assert "CUDA" in content
    assert "Python 3.9" not in content
    assert "--index-url" not in content


def test_readme_is_self_contained_and_documents_public_workflow_boundaries() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="ascii")
    normalized = " ".join(readme.split())
    required = (
        "BrainVLM-Midline is one internal Diagnostic Sub-Agent model",
        "modified training and inference source",
        "does not call another source checkout",
        "30 selected 2D MRI slices",
        "No diagnostic labels or results are bundled",
        "midline-predict-tumor",
        "midline-predict-who-grade",
        "midline-extract-features",
        "midline-train-attention-mil",
        "midline-train-brainvlm",
        "midline-run",
        "midline_agent.survival",
        "EvidenceProvider",
        "outside the Git checkout",
        "official OpenAI Responses API",
        "does not reproduce the internal treatment report quality",
    )
    assert all(item in normalized for item in required)
    assert "git clone" not in readme
    assert "--" + "source-root" not in readme
    assert "BRAINVLM" + "_SOURCE" not in readme
    assert "PACKAGE.MODULE:FACTORY" not in readme
    assert "The reference labels for BT0156" not in readme
    assert "/secure/" not in readme


def test_published_orchestrator_examples_supply_the_positional_llm_client() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="ascii")
    provider_docs = (PROJECT_ROOT / "docs" / "treatment-providers.md").read_text(
        encoding="ascii"
    )
    normalized_readme = " ".join(readme.split())
    normalized_provider_docs = " ".join(provider_docs.split())

    assert "MidlineOrchestrator(None, survival_workflow=workflow)" in readme
    assert "no Treatment LLM client" in normalized_readme
    assert "Treatment is recorded as unavailable" in normalized_readme

    treatment_example = (
        'client = OpenAiResponsesClient(api_key=os.environ["OPENAI_API_KEY"], '
        'model=os.environ["OPENAI_MODEL"])'
    )
    orchestrator_example = (
        "orchestrator = MidlineOrchestrator( client, treatment_providers=(provider,), )"
    )
    for content in (normalized_readme, normalized_provider_docs):
        assert "from midline_agent.orchestrator import MidlineOrchestrator" in content
        assert (
            "from midline_agent.treatment.client import OpenAiResponsesClient" in content
        )
        assert treatment_example in content
        assert orchestrator_example in content

    for path in PROJECT_ROOT.rglob("*.md"):
        content = path.read_text(encoding="ascii")
        for match in re.finditer(r"MidlineOrchestrator\(\s*([^,\n)]*)", content):
            first_argument = match.group(1)
            assert "=" not in first_argument, (
                f"{path.relative_to(PROJECT_ROOT)} omits the positional llm_client: "
                f"{match.group(0)}"
            )


def test_brainvlm_runtime_documentation_records_assets_schema_and_scope() -> None:
    documentation = (PROJECT_ROOT / "docs" / "brainvlm-midline.md").read_text(
        encoding="ascii"
    )
    required = (
        "Llama 3.1-8B Instruct",
        "BiomedCLIP",
        "Diffuse midline glioma, H3 K27-altered",
        "Paediatric-type diffuse glioma (checkpoint superclass)",
        "Glioblastoma, IDH-wildtype",
        "Astrocytoma, IDH-mutant",
        "Oligodendroglioma, IDH-mutant and 1p/19q-codeleted",
        "H3K27M-mutated midline glioma",
        "Pediatric-type diffuse glioma",
        "IDH-mutated astrocytoma",
        "patient_id",
        "image_path",
        "modality",
        "slice_index",
        "who_class",
        "WHO_class",
        "who_grade",
        "hidden_and_logits",
        "logit_dimension=5",
        "hidden_dimension=4096",
        "configs/brainvlm_tumor.example.yaml",
        "configs/brainvlm_who.example.yaml",
        "0e3eff840fb5bd5eb9be0ef27b03d17f2fb7aa1aa9dca248300a1f23d7bfab18",
        "96c6637b1ae41edaaccc0a32e8b845abeb3995eec232cdcc48e1507b152bc0ca",
        "bd3f7b8d5606e649adef490dddb2ec85e37ed1ab8dfdc4304dc185f9723d4eca",
        "Intentional exclusions",
    )
    assert all(item in documentation for item in required)
    assert "git clone" not in documentation
    assert "--" + "source-root" not in documentation
    assert "BRAINVLM" + "_SOURCE" not in documentation
    assert "/userhome/" not in documentation


def test_obsolete_external_checkout_documents_are_absent() -> None:
    obsolete = (
        "docs/external-brainvlm-adapter.md",
        "docs/superpowers/specs/2026-11-27-checkpoint-inference-design.md",
        "docs/superpowers/plans/2026-11-27-checkpoint-inference.md",
    )
    assert all(not (PROJECT_ROOT / path).exists() for path in obsolete)


def test_notices_and_licenses_record_precise_upstream_attribution() -> None:
    notices = (PROJECT_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="ascii")
    normalized_notices = " ".join(notices.split())
    required = (
        "MiniGPT-4",
        "Deyao Zhu",
        "LAVIS",
        "Salesforce, Inc.",
        "Transformers",
        "PEFT",
        "OpenCLIP",
        "BiomedCLIP",
        "Llama 3.1",
        "subject to its own license",
        "related BrainVLM work",
        "https://github.com/HKU-HealthAI/BrainVLM",
        "HKU-HealthAI",
        "A Vision-Language Foundation Model for Comprehensive Brain Tumor Diagnosis from Preoperative Multimodal Data",
        "BrainVLM licensing is not inferred",
        "No code expression from BrainVLM is included",
        "No comparison code is included",
    )
    assert all(item in normalized_notices for item in required)
    assert "external runtime" not in notices

    mini = (PROJECT_ROOT / "licenses" / "BSD-3-Clause-MiniGPT4.txt").read_text(
        encoding="ascii"
    )
    lavis = (PROJECT_ROOT / "licenses" / "BSD-3-Clause-LAVIS.txt").read_text(
        encoding="ascii"
    )
    apache = (PROJECT_ROOT / "licenses" / "Apache-2.0.txt").read_text(encoding="ascii")
    assert "Copyright 2023 Deyao Zhu" in mini
    assert "Copyright (c) 2022 Salesforce, Inc." in lavis
    assert "Apache License" in apache and "END OF TERMS AND CONDITIONS" in apache
    assert "Copyright 2018- The Hugging Face team" in apache


def test_provenance_inventory_covers_final_derived_files_with_current_digests() -> None:
    provenance = (PROJECT_ROOT / "docs/provenance/brainvlm-midline-sources.md").read_text(
        encoding="ascii"
    )
    digest_rows = dict(
        re.findall(
            r"^\| `([^`]+\.py)` \| `([0-9a-f]{64})` \|",
            provenance,
            flags=re.MULTILINE,
        )
    )
    required_paths = {
        "midline_agent/diagnostic/brainvlm_midline/__init__.py",
        "midline_agent/diagnostic/brainvlm_midline/collate.py",
        "midline_agent/diagnostic/brainvlm_midline/config.py",
        "midline_agent/diagnostic/brainvlm_midline/datasets.py",
        "midline_agent/diagnostic/brainvlm_midline/llama.py",
        "midline_agent/diagnostic/brainvlm_midline/model.py",
        "midline_agent/diagnostic/brainvlm_midline/prompts.py",
        "midline_agent/diagnostic/brainvlm_midline/training.py",
        "midline_agent/diagnostic/brainvlm_midline/vision.py",
        "midline_agent/diagnostic/brainvlm_runtime.py",
        "midline_agent/diagnostic/feature_store.py",
        "scripts/extract_features.py",
        "scripts/train_brainvlm_midline.py",
    }
    assert required_paths <= digest_rows.keys()
    for relative_path in required_paths:
        actual = hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()
        assert digest_rows[relative_path] == actual
    assert "no defensible local source revision" in provenance
    assert "does not invent a Git lineage" in " ".join(provenance.split())
    required_attribution = (
        "https://github.com/Vision-CAIR/MiniGPT-4",
        "https://github.com/salesforce/LAVIS",
        "https://github.com/HKU-HealthAI/BrainVLM",
        "HKU-HealthAI",
        "No license is inferred for BrainVLM",
        "No code expression from BrainVLM is included",
        "project-specific modifications have no defensible Git lineage or independently established license",
        "only behavior and checkpoint observations",
        "clean MIT reimplementation",
        "Known upstream BSD-3-Clause material",
        "uncertain local code is excluded",
    )
    assert all(item in " ".join(provenance.split()) for item in required_attribution)

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="ascii")
    assert "https://github.com/HKU-HealthAI/BrainVLM" in readme


def test_env_template_exposes_only_official_openai_configuration() -> None:
    variables = (PROJECT_ROOT / ".env.example").read_text(encoding="ascii").splitlines()
    assert variables == ["OPENAI_API_KEY=", "OPENAI_MODEL="]
    joined = "\n".join(variables)
    assert "OPENAI_BASE" + "_URL" not in joined


def test_public_documentation_does_not_publish_private_server_paths() -> None:
    documents = (
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "THIRD_PARTY_NOTICES.md",
        PROJECT_ROOT / "docs/brainvlm-midline.md",
        PROJECT_ROOT / "docs/treatment-providers.md",
        PROJECT_ROOT / "docs/superpowers/plans/2026-11-27-treatment-private-evidence.md",
        PROJECT_ROOT / "docs/superpowers/specs/2026-11-27-treatment-private-evidence-design.md",
    )
    forbidden = ("/secure/", "/userhome/", "/home/", "C:\\Users\\")
    for path in documents:
        content = path.read_text(encoding="ascii")
        assert not any(item in content for item in forbidden), path
