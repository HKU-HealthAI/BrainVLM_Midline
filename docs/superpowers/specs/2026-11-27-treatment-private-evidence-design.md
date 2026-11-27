# Treatment Private-Evidence Boundary Design

## Objective

Document how the public Treatment Sub-Agent can use deployment-owned evidence
without publishing the internal case database, local knowledge base, physician
rules, credentials, or derived records.

## Public Release Boundary

The repository publishes the Treatment Sub-Agent, the official OpenAI Responses
API client, structured report validation, and the `EvidenceProvider` protocol.
It does not publish an internal case database, local knowledge corpus, physician
rules, database schema, connection details, cached search results, or generated
reports.

Private evidence must remain outside the Git checkout in a caller-owned,
deployment-controlled location. No filesystem layout is implied. Access control,
encryption, retention, backups, and privacy review remain deployment
responsibilities.

## Integration Contract

A deployment implements the existing `EvidenceProvider` protocol and converts
authorized retrieval results into a fully materialized tuple of `EvidenceItem`
values. The provider is passed to
`MidlineOrchestrator(treatment_providers=(provider,))`. The public package does
not inspect the private directory or infer a database format.

The integrated `midline-run` CLI does not automatically discover or load a
private database. A deployment that needs private evidence must construct the
orchestrator in deployment-owned code. This limitation must be stated directly
in the README and provider documentation.

## Runtime Claim

With `OPENAI_API_KEY`, `OPENAI_MODEL`, a configured Diagnostic workflow, and an
injected provider, the public orchestration path can collect evidence, request a
structured Treatment report through the official OpenAI Responses API, validate
all cited source IDs, and return a `CaseState`.

The returned in-memory state is sensitive: it retains the provider-supplied
`EvidenceItem` values, and the rendered report can include source names and
URLs. Deployment code must call `CaseState.to_trace()` and persist only the
resulting privacy-minimized `CaseTrace` when a trace is required. It must not
serialize or persist the complete `CaseState` without a separate privacy review.

The public release can validate this interface with synthetic in-memory test
providers. It cannot demonstrate the private database retrieval or reproduce
the report quality obtained with the unavailable internal evidence.

## Evidence Exposure and Provider Trust

The current public boundary sends selected evidence metadata to the model and
does not send raw database excerpts or URLs. It sends provider-issued
`source_id` values verbatim and does not impose a maximum evidence count.
Deployments must therefore issue non-identifying opaque source IDs, limit
retrieval results before returning them, and review all remaining metadata for
disclosure risk. The public implementation does not itself guarantee that
provider metadata is privacy-safe.

Because the model does not receive evidence content, the public implementation
does not reproduce the full private evidence-grounding behavior. Documentation
must not claim otherwise.

Providers are trusted in-process code. The collector applies a bounded wait and
continues after most provider failures, but it does not sandbox providers or
terminate a provider thread after a timeout. Database access controls and
credential handling must be enforced by the deployment.

## Documentation Changes

The README introduction, release boundary, and Treatment section will:

- state that no internal database or private evidence is distributed;
- direct deployments to keep private evidence outside the Git checkout;
- identify `EvidenceProvider` and `MidlineOrchestrator(treatment_providers=...)`
  as the supported injection boundary;
- state that `midline-run` does not automatically load private evidence; and
- distinguish interface-level operability from reproduction of private-data
  report quality and full private evidence-grounding behavior.

Top-level statements about evidence-grounded reports and complete source code
will be qualified so they refer to the public orchestration and validation
implementation, not the undistributed private data, schema, or retrieval code.

`docs/treatment-providers.md` will provide the same boundary in more detail. It
will describe providers as trusted in-process components with bounded waits,
not isolated or sandboxed components. It will not include private schemas,
records, example patient data, or a generic database loader.

## Verification

Verification will add an integration test that injects a synthetic provider
through `MidlineOrchestrator(treatment_providers=...)`, confirms that the full
in-memory result retains its evidence, and confirms that `CaseState.to_trace()`
omits evidence excerpts and URLs. A CLI test will confirm that the public CLI
has no private-provider discovery or database-path option.

Final verification will run the focused Treatment, provider, client, report,
orchestrator, and CLI tests; the complete test suite;
`python scripts/release_audit.py --root . --manifest release-manifest.txt`;
`python scripts/release_audit.py --root . --tracked-only`; an exact comparison
of tracked files against `release-manifest.txt`; the Han-character scan; and
`git diff --check`.
