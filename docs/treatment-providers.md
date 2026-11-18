# Treatment Providers

The public Treatment Agent has no bundled guideline corpus, local case
database, web cache, or evidence source. Supply physician-authored skills,
user-owned local knowledge, and user-owned similar-case providers through the
`EvidenceProvider` protocol. Providers are isolated and bounded by a caller
configured deadline; a failed provider does not erase evidence from other
providers. A provider must return a fully materialized tuple of validated
`EvidenceItem` values. Lazy iterables are rejected without iteration.

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
Persisted `CaseTrace` values omit excerpts and URLs, so traces should still
undergo local privacy review before retention.
