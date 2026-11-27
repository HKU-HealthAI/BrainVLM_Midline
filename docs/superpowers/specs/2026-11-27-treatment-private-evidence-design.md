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

Private evidence must remain outside the Git checkout in a deployment-controlled
directory, for example `/secure/midlineagent/treatment_evidence/`. This path is
illustrative rather than a required filesystem layout. Access control,
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
all cited source IDs, and return a privacy-minimized trace.

The public release can validate this interface with synthetic in-memory test
providers. It cannot demonstrate the private database retrieval or reproduce
the report quality obtained with the unavailable internal evidence.

## Evidence Exposure

The current public boundary sends bounded evidence metadata to the model and
does not send raw database excerpts or URLs. This preserves the release's
privacy boundary but means the public implementation does not reproduce the
full private evidence-grounding behavior. Documentation must not claim
otherwise.

## Documentation Changes

The README Treatment section will:

- state that no internal database or private evidence is distributed;
- direct deployments to keep private evidence outside the Git checkout;
- identify `EvidenceProvider` and `MidlineOrchestrator(treatment_providers=...)`
  as the supported injection boundary;
- state that `midline-run` does not automatically load private evidence; and
- distinguish interface-level operability from reproduction of private-data
  report quality.

`docs/treatment-providers.md` will provide the same boundary in more detail,
without including private schemas, records, example patient data, or a generic
database loader.

## Verification

The change is documentation-only. Verification will run the Treatment,
provider, client, report, and orchestrator tests; the complete test suite; both
release-audit modes; the release manifest comparison; the Han-character scan;
and Git whitespace checks.
