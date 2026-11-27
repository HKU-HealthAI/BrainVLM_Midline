# Third-Party Notices

MidlineAgent is distributed under the repository license. The notices below
identify source work, installed libraries, model assets, and attributed data
that remain subject to their own terms. Model weights and third-party Python
packages are not relicensed by this repository.

## BrainVLM-related implementation

The internal BrainVLM-Midline model is a clean MIT reimplementation of task
behavior and checkpoint contracts. It cites the HKU-HealthAI
[BrainVLM repository](https://github.com/HKU-HealthAI/BrainVLM), *A
Vision-Language Foundation Model for Comprehensive Brain Tumor Diagnosis from
Preoperative Multimodal Data* by Yinong Wang et al., as related work. That
repository states that BrainVLM builds on MiniGPT-4. BrainVLM licensing is not
inferred because no license was identified at the repository root during the
source review. No code expression from BrainVLM is included. Only behavior and
checkpoint observations from the related BrainVLM work informed the
reimplementation.

MidlineAgent ships its own focused training and inference source and does not
require any of these project source trees to run.

[MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) is BSD-3-Clause
software, Copyright 2023 Deyao Zhu. The complete notice is in
`licenses/BSD-3-Clause-MiniGPT4.txt`.

[LAVIS](https://github.com/salesforce/LAVIS) is BSD-3-Clause software,
Copyright (c) 2022 Salesforce, Inc. The complete notice is in
`licenses/BSD-3-Clause-LAVIS.txt`.

The source ownership inventory, immutable local-source digests, inspected
upstream revisions, modifications, and final released-file digests are in
`docs/provenance/brainvlm-midline-sources.md`. No comparison code is included.

## Installed model libraries

The diagnostic package installs, rather than vendors, PyTorch, torchvision,
Transformers, PEFT, OpenCLIP, OmegaConf, einops, Pillow, NumPy, safetensors,
and PyYAML.

- Transformers and PEFT are Hugging Face projects distributed under Apache
  License 2.0. The complete Transformers license and copyright notice,
  `Copyright 2018- The Hugging Face team`, are reproduced in
  `licenses/Apache-2.0.txt`.
- OpenCLIP is distributed under the MIT License and carries its upstream
  contributor copyright notice in its installed distribution.
- BiomedCLIP is accessed as the Microsoft model identified by
  `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` through
  OpenCLIP. Its model files remain subject to the model card and associated
  terms.
- Llama 3.1-8B Instruct is a separately obtained Meta model and is subject to
  its own license and model-access terms. It is not distributed here.

The Survival and Treatment extras similarly install their declared libraries
from their own distributions. The official OpenAI Python SDK is used only with
the official OpenAI Responses API.

## UTSW-Glioma test case

The files under `test_data/utsw_glioma_bt0156/images/` are modified, selected
2D PNG derivatives from the UTSW-Glioma collection. The source citation is
Reddy et al. (2026), *The University of Texas Southwestern Glioma MRI dataset
with molecular marker characterization and segmentations (UTSW-Glioma)*, The
Cancer Imaging Archive, https://doi.org/10.7937/DFAE-1B86.

The collection is available at
https://www.cancerimagingarchive.net/collection/utsw-glioma/ under CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/. The case-level
`test_data/utsw_glioma_bt0156/ATTRIBUTION.md` contains the complete citation
and modification notice.
