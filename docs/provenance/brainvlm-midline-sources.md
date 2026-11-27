# BrainVLM-Midline Source Provenance

## Purpose

This inventory records the source material inspected while creating the
self-contained BrainVLM-Midline implementation. It is a pre-copy ownership
gate: no non-trivial source text may enter the release unless its origin,
copyright holder, license, and modifications are recorded here first.

The local research tree at `/projects/MiniGPT-4` is not an independent Git
checkout. Its containing repository reports that directory as untracked, so
there is no defensible local source revision. This inventory therefore records
immutable SHA-256 digests for the exact local files and does not invent a Git
lineage. The official MiniGPT-4 repository was independently inspected at
revision `d94738a7626ec43eba6c2cddf3cd2043f1a9689a`; that revision is an
upstream reference, not a claim that the modified local tree exactly descends
from it. The official LAVIS repository was independently inspected at revision
`05cb670f47296522e24e7727e366b4a997d28582`.

## Ownership Rule

Project-specific prompts, task behavior, state names, and workflow contracts
are reimplemented from documented behavior and checkpoint observations. New
implementation text authored for this repository is covered by the repository
MIT license and attributed to MidlineAgent contributors. It is not a verbatim
copy of the historical research files.

If numerical parity later requires retained source text from MiniGPT-4 or
LAVIS, the released file must carry a prominent modification notice and the
applicable BSD-3-Clause notice before commit. If parity requires retained code
from the local Llama implementation, the released file must carry its original
Hugging Face and EleutherAI copyright header, an Apache-2.0 modification
notice, and a completed inventory row before commit. Material with uncertain
ownership or an unidentified license must be excluded and reimplemented from
behavior instead.

## Local Source Digests

| Local source path | SHA-256 | Declared origin and license | Use in release |
| --- | --- | --- | --- |
| `minigpt4/models/minigpt4.py` | `4688da6ff638ba6160e3c8fdbd61a2a9ea47dc9d46bb030a3475f01d3330c4ae` | Modified MiniGPT-4 project source; upstream project is BSD-3-Clause, copyright 2023 Deyao Zhu | Architecture and state-layout behavior only |
| `minigpt4/models/minigpt_base.py` | `fc6ce17024e9c5ee602b11bdf65d38f11866b9a5750c10b706936817ee786bd8` | Modified MiniGPT-4 project source; upstream project is BSD-3-Clause, copyright 2023 Deyao Zhu | Prompt wrapping, supervised input, and generation behavior only |
| `minigpt4/models/base_model.py` | `1ec55a1c764bf45244ac34f7dfc91c332c04a24d2300f350a197612d5a8ff907` | LAVIS-derived source; BSD-3-Clause, copyright 2022 Salesforce, Inc. | Autocast and model utility behavior only |
| `minigpt4/models/modeling_llama.py` | `04e3d6636b33534e72443c3387eaa490b2821f252f9c4cebbffc1b9ad9565af9` | Local adaptation around the Llama implementation below | Loss-reduction behavior only; no planned source copy |
| `minigpt4/models/modelling_llama_source.py` | `8df5eae8ed5d3219a1b8e894fc587fe75741c80f2dce1b7c68ec29bd06103219` | Hugging Face and EleutherAI-derived source; Apache-2.0 | Potential parity fallback only; excluded unless the gate above is completed |
| `minigpt4/datasets/datasets/caption_datasets.py` | `1dd4fffbf27966d7f1d26b524226b183e4757b8b36ef1d7a44d77a15288e4d0d` | LAVIS-derived file with project modifications; BSD-3-Clause, copyright 2022 Salesforce, Inc. | Tumor and WHO dataset behavior only |
| `minigpt4/common/optims.py` | `0b6dc7e70464a24c47594598298cbbae516557053855bc1a0cc17b78f92a86d3` | LAVIS-derived source; BSD-3-Clause, copyright 2022 Salesforce, Inc. | Scheduler behavior only |
| `minigpt4/runners/runner_base.py` | `bb1377f8910bd7e8f3c7621702ffdb53f29ec2531333072e53fa5209dbeef402` | LAVIS-derived source with project modifications; BSD-3-Clause, copyright 2022 Salesforce, Inc. | Optimizer, checkpoint, and resume behavior only |
| `minigpt4/tasks/base_task.py` | `2050b156d642cf253f4d1761a66495141efe3adbde927f68d36dde764cf61904` | LAVIS-derived source with project modifications; BSD-3-Clause, copyright 2022 Salesforce, Inc. | AMP and accumulation behavior only |
| `train.py` | `6bf702df2188215a32d071137fb0be12fce0fb957bd265fb43cbe8809413b453` | LAVIS-derived source with project modifications; BSD-3-Clause, copyright 2022 Salesforce, Inc. | Seed and distributed-launch behavior only |

## Planned Release Mapping

| Released path | Source material | Planned modification | Holder and release license |
| --- | --- | --- | --- |
| `midline_agent/diagnostic/brainvlm_midline/config.py` | Approved public design and checkpoint manifest | New strict dataclasses and validators | MidlineAgent contributors; MIT |
| `midline_agent/diagnostic/brainvlm_midline/prompts.py` | Historical runtime prompts and the two dataset classes in `caption_datasets.py` | Clean English-only canonical prompt and label implementation | MidlineAgent contributors; MIT, with MiniGPT-4 and LAVIS acknowledged |
| `midline_agent/diagnostic/brainvlm_midline/vision.py` | `minigpt4.py` and dataset preprocessing behavior | Clean dependency-injected OpenCLIP implementation | MidlineAgent contributors; MIT, with MiniGPT-4 and LAVIS acknowledged |
| `midline_agent/diagnostic/brainvlm_midline/llama.py` | `minigpt_base.py`, `modeling_llama.py`, and installed Transformers 4.46.3 APIs | Clean wrapper around installed Transformers and PEFT | MidlineAgent contributors; MIT; Transformers remains an Apache-2.0 dependency |
| `midline_agent/diagnostic/brainvlm_midline/model.py` | `minigpt4.py` and `minigpt_base.py` behavior | Clean focused model assembly preserving checkpoint names | MidlineAgent contributors; MIT, with MiniGPT-4 acknowledged |
| `midline_agent/diagnostic/brainvlm_midline/datasets.py` | `tcia_classification_who` and `just_who_grade` behavior | Clean strict parser with historical silent fallbacks removed | MidlineAgent contributors; MIT, with LAVIS acknowledged |
| `midline_agent/diagnostic/brainvlm_midline/collate.py` | Valid sample shapes observed in the dataset classes | New strict collator | MidlineAgent contributors; MIT |
| `midline_agent/diagnostic/brainvlm_midline/training.py` | LAVIS optimizer, scheduler, AMP, and checkpoint behavior | Clean focused training implementation without historical integrations | MidlineAgent contributors; MIT, with LAVIS acknowledged |
| `midline_agent/diagnostic/brainvlm_runtime.py` | Existing MidlineAgent bridge and approved internal contract | Replace external loading with direct internal construction | MidlineAgent contributors; MIT |
| `scripts/train_brainvlm_midline.py` | Approved training contract | New focused command | MidlineAgent contributors; MIT |

Final released-file SHA-256 values and any changed derivation status will be
added after implementation. A non-trivial copied block changes a row from
clean reimplementation to derived source and triggers the corresponding
file-header and license requirements before commit.

### Implemented Release Digests

| Released path | SHA-256 | Derivation status |
| --- | --- | --- |
| `midline_agent/diagnostic/brainvlm_midline/__init__.py` | `7c979b8c88c8c6a36b9075d3af15b499c6489570d396b92bc043904203f4dfc1` | New package export; MIT |
| `midline_agent/diagnostic/brainvlm_midline/config.py` | `8650e78ee32f82242fc4689d35c9427ebf64e4dd1d0dd8bc3a2acfb342cc8978` | Clean behavioral reimplementation; MIT |
| `midline_agent/diagnostic/brainvlm_midline/prompts.py` | `3e3c5c9aa8195d585ecb0e76511621b9ff296f840a7da37dc890f2835b43f1be` | Clean behavioral reimplementation; MIT |
| `midline_agent/diagnostic/brainvlm_midline/vision.py` | `ba87b977ee6eae8c56ea2eb9bc385ff8c75c6d9ef6de29a48415410dd928a8b9` | Clean behavioral reimplementation; MIT |
| `midline_agent/diagnostic/brainvlm_midline/llama.py` | `9eb4265e30cd4fc811bfacf9a787e2e12590836d628e09b009a89569cadfeec3` | Clean wrapper around installed Transformers and PEFT; MIT |
| `midline_agent/diagnostic/brainvlm_midline/model.py` | `99c80b08c752b6a9414901ebddac44416dc60074cd49289dc225d9ec84f963be` | Clean behavioral reimplementation; MIT |
| `midline_agent/diagnostic/checkpoints.py` | `79ad8650a061498d6f4075721f9d7bf1aa7b412189125bc26955bdaae07380a2` | Existing MIT implementation extended with strict BrainVLM-Midline loading |
| `midline_agent/diagnostic/brainvlm_runtime.py` | `93577ef1a5e71ba84ce820d3c738af103a7526973f460efc4fc7c6dcfef4e28e` | Internal model runtime replacing source-checkout loading; MIT |

## License Files

- `licenses/BSD-3-Clause-MiniGPT4.txt` contains the official MiniGPT-4
  BSD-3-Clause terms and copyright notice.
- `licenses/BSD-3-Clause-LAVIS.txt` contains the official LAVIS
  BSD-3-Clause terms and copyright notice.
- `licenses/Apache-2.0.txt` contains the complete Apache License 2.0 used by
  Transformers and the potential Llama fallback.
