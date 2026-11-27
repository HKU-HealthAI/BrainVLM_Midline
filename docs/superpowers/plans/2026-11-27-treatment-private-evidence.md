# Treatment Private-Evidence Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accurately document and test how a deployment can inject a non-public internal evidence source into the Treatment Sub-Agent without placing private data in the public repository.

**Architecture:** Keep the existing `EvidenceProvider` and `MidlineOrchestrator` runtime interfaces unchanged. Add characterization tests for provider injection, sensitive in-memory state, minimized persisted traces, and the absence of private-database CLI discovery; then align the README and provider documentation with those verified boundaries.

**Tech Stack:** Python 3.9+, pytest, Markdown, existing MidlineAgent schemas and release-audit tooling.

---

### Task 0: Commit the Reviewed Plan

**Files:**
- Create: `docs/superpowers/plans/2026-11-27-treatment-private-evidence.md`
- Modify: `release-manifest.txt`

- [ ] **Step 1: Verify and commit the plan metadata**

```bash
python scripts/release_audit.py --root . --manifest release-manifest.txt
git diff --check
git add docs/superpowers/plans/2026-11-27-treatment-private-evidence.md release-manifest.txt
git commit -m "docs: plan private treatment evidence documentation"
```

Expected: the audit exits zero and the plan and manifest are committed before
implementation begins.

### Task 1: Lock the Public Integration Contract in Tests

**Files:**
- Modify: `tests/test_orchestrator.py`
- Modify: `tests/test_cli_entrypoints.py`

- [ ] **Step 1: Add the orchestrator provider-injection test**

Import `EvidenceItem`, `to_jsonable`, and `StaticEvidenceProvider`, then add a
test that injects an opaque private source through the public orchestrator:

```python
def test_injected_treatment_evidence_is_retained_in_memory_but_minimized_in_trace() -> None:
    source = EvidenceItem(
        "opaque-source-1",
        "similar_case",
        excerpt="PRIVATE_EVIDENCE_SENTINEL",
        url="https://private.invalid/source",
        source_name="Deployment-owned source",
        retrieval_status="verified",
        applicability="Relevant to the supplied tumor context.",
    )

    class EvidenceClient(SuccessfulClient):
        def generate(self, request):
            response = super().generate(request)
            if request.schema_name == "midline_treatment_report":
                response.parsed["evidence"] = ["opaque-source-1"]
            return response

    state = MidlineOrchestrator(
        EvidenceClient(),
        diagnostic_workflow=lambda case: DiagnosticResult.unavailable("checkpoint unavailable"),
        treatment_providers=(StaticEvidenceProvider("private", (source,)),),
    ).run(CaseInput(case_id="opaque-case"))

    assert state.treatment.status.state == "completed"
    assert state.treatment.evidence == (source,)
    assert "PRIVATE_EVIDENCE_SENTINEL" in state.treatment.evidence[0].excerpt
    serialized_trace = str(to_jsonable(state.to_trace()))
    assert "PRIVATE_EVIDENCE_SENTINEL" not in serialized_trace
    assert "private.invalid" not in serialized_trace
```

Use a mutable response object or construct a dedicated response so that the
test does not mutate an immutable mapping.

- [ ] **Step 2: Add the CLI non-discovery test**

Extend `test_pipeline_runner_exposes_no_custom_openai_endpoint` or add a focused
test that runs `scripts/run_pipeline.py --help` and asserts the public CLI does
not expose `--internal-case-db`, `--private-evidence-root`, or
`--treatment-provider`.

- [ ] **Step 3: Run the focused tests**

Run:

```bash
pytest -q tests/test_orchestrator.py tests/test_cli_entrypoints.py
```

Expected: the runtime characterization tests pass against the existing code.
If a test fails, investigate the actual contract before changing production
code; this task does not authorize a private database loader.

- [ ] **Step 4: Commit the tests**

```bash
git add tests/test_orchestrator.py tests/test_cli_entrypoints.py
git commit -m "test: define private treatment evidence boundary"
```

### Task 2: Correct the Public Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/treatment-providers.md`

- [ ] **Step 1: Add documentation assertions**

Extend a README/provider-document test in `tests/test_cli_entrypoints.py` to
require these concepts in the public documentation:

```python
assert "not distributed" in readme
assert "outside the Git checkout" in readme
assert "EvidenceProvider" in readme
assert "treatment_providers" in readme
assert "does not automatically" in readme
assert "CaseState.to_trace()" in provider_docs
assert "trusted in-process" in provider_docs
assert "does not sandbox" in provider_docs
assert "outside the Git checkout" in provider_docs
assert "EvidenceProvider" in provider_docs
assert "treatment_providers" in provider_docs
assert "does not automatically" in provider_docs
```

Run the focused test and expect failure before editing the documentation.

- [ ] **Step 2: Qualify the README introduction and release boundary**

Change the top-level description so the Treatment Sub-Agent is described as a
structured treatment-report workflow with injectable evidence providers. State
that the release contains complete public orchestration, validation, and
Sub-Agent code, but excludes the internal case database, private knowledge
sources, their schemas, private retrieval implementation, and generated
reports.

- [ ] **Step 3: Document private evidence deployment in the Treatment section**

State that private evidence must remain outside the Git checkout in a
caller-owned, deployment-controlled location. Explain that deployment-owned code
must implement `EvidenceProvider`, return a bounded tuple of `EvidenceItem`
values with non-identifying opaque source IDs, and pass the provider through
`MidlineOrchestrator(treatment_providers=(provider,))`. State explicitly that
`midline-run` does not automatically discover a private database.

Clarify that the public package validates interface-level operation but cannot
reproduce private database retrieval or private-data report quality. Do not add
a database schema, patient record, sample private data, generic loader, or path
inside the Git checkout.

- [ ] **Step 4: Correct provider security and persistence language**

Update `docs/treatment-providers.md` to state that providers are trusted
in-process components. A timeout bounds how long the collector waits but does
not terminate or sandbox a provider thread. Document that the complete
`CaseState` retains evidence and report metadata; deployments must call
`CaseState.to_trace()` and persist only the minimized `CaseTrace` unless a
separate privacy review authorizes other storage.

Document that raw excerpts and URLs are not sent to the model, while source IDs
are sent verbatim. Require deployment-issued non-identifying IDs, bounded
results, and privacy review of all metadata.

Repeat the deployment boundary in this detailed provider document: private
evidence belongs outside the Git checkout; deployment-owned code implements
`EvidenceProvider` and injects it with
`MidlineOrchestrator(treatment_providers=(provider,))`; and `midline-run` does
not automatically discover or load a private database.

- [ ] **Step 5: Run focused documentation and Treatment tests**

Run:

```bash
pytest -q \
  tests/test_cli_entrypoints.py \
  tests/test_orchestrator.py \
  tests/test_treatment_agent.py \
  tests/test_treatment_client.py \
  tests/test_treatment_providers.py \
  tests/test_treatment_report.py
```

Expected: all focused tests pass.

- [ ] **Step 6: Commit the documentation**

```bash
git add README.md docs/treatment-providers.md tests/test_cli_entrypoints.py
git commit -m "docs: explain private treatment evidence deployment"
```

### Task 3: Verify and Publish the Release Update

**Files:**
- Modify: `release-manifest.txt` only if a new tracked file is not already listed

- [ ] **Step 1: Run the complete test suite**

Run `pytest -q`.

Expected: all tests pass with no failures.

- [ ] **Step 2: Run both release audits**

```bash
python scripts/release_audit.py --root . --manifest release-manifest.txt
python scripts/release_audit.py --root . --tracked-only
```

Expected: both commands exit zero and scan every candidate file.

- [ ] **Step 3: Compare the manifest and scan publication boundaries**

```bash
comm -3 <(git ls-files | sort) <(sort release-manifest.txt)
rg -n --pcre2 '\p{Han}' .
git ls-files | rg -i '(^|/)(\.env$|checkpoints?(/|$)|results?(/|$)|predictions?(/|$)|.*\.(ckpt|pth|pt|safetensors|log)$)'
git diff --check
git diff --check origin/main...HEAD
git status --short --branch
```

Expected: the comparison, Han scan, prohibited-artifact scan, and whitespace
check produce no findings; Git reports only the intended commits.

- [ ] **Step 4: Push the verified fast-forward update**

Run:

```bash
git fetch origin
git merge-base --is-ancestor origin/main HEAD
git push -u origin feature/checkpoint-inference
git push origin HEAD:main
git ls-remote origin refs/heads/main refs/heads/feature/checkpoint-inference
```

Expected: the ancestry check exits zero, both pushes are fast-forward updates,
and both remote refs resolve to the locally tested `HEAD` commit.
