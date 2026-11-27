# Treatment Providers

The public Treatment Agent has no bundled guideline corpus, local case
database, web cache, or evidence source. Supply physician-authored skills,
user-owned local knowledge, and user-owned similar-case providers through the
`EvidenceProvider` protocol. The internal case database, private schemas,
physician-authored rules, deployment retrieval code, credentials, and derived
records are not distributed.

Private evidence must remain outside the Git checkout in a
deployment-controlled location, for example
`/secure/midlineagent/treatment_evidence/`. The path is illustrative rather
than required. The public package does not inspect that directory, infer a
private database format, or provide a generic database loader.

A deployment implements `EvidenceProvider`, converts authorized retrieval
results into a fully materialized tuple of validated `EvidenceItem` values, and
passes the provider through
`MidlineOrchestrator(treatment_providers=(provider,))`. Lazy iterables are
rejected without iteration. The public `midline-run` command
does not automatically discover or load a private database; deployment-owned
code must construct the orchestrator when private evidence is required.

Providers are trusted in-process components. The collector applies a
caller-configured deadline and continues after most provider failures, but the
deadline only bounds how long the collector waits. It does not sandbox a
provider or terminate its daemon thread after a timeout. Deployments remain
responsible for database permissions, credential handling, network isolation,
result limits, and provider lifecycle management.

The model receives selected evidence metadata rather than raw database excerpts
or URLs. Provider-issued source IDs are sent verbatim. Providers must therefore
use non-identifying opaque source IDs, return a bounded number of results, and
review all remaining metadata for disclosure risk. This public boundary
validates provider integration and cited source IDs; it does not reproduce the
full private evidence-grounding behavior or private-data report quality.

The optional `OpenAiWebSearchProvider` uses the official OpenAI Responses API
web-search tool at deployment time. It stores only injected `EvidenceItem`
values in memory and ships no provider data.

`OpenAiResponsesClient` accepts `OPENAI_API_KEY` and `OPENAI_MODEL` values from
the deployment environment. It uses structured Responses output with
`store=false`, a request timeout, and bounded retries. Only the official
OpenAI endpoint is supported.

Reports validate every cited source ID against collected evidence before
rendering Markdown. Model text and evidence metadata are escaped as plain text;
only validated, query-free HTTPS evidence URLs are rendered as URLs. When no
verified evidence is available, the report states that limitation explicitly.

The complete in-memory `CaseState` retains provider evidence, and rendered
report Markdown can include source names and URLs. Treat both as sensitive.
Before persistence, call `CaseState.to_trace()` and retain only the resulting
privacy-minimized `CaseTrace`, unless a separate privacy review authorizes other
storage. The minimized trace omits evidence excerpts and URLs, but it should
still undergo local privacy review before retention.
