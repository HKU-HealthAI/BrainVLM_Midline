# Third-Party Notices

This repository does not vendor third-party model source, model weights,
evidence corpora, or generated artifacts. It includes 30 selected 2D PNG
derivatives from one attributed UTSW-Glioma case for checkpoint verification.

The optional runtime dependencies are NumPy, pandas, Pillow, PyTorch,
safetensors, scikit-learn, scikit-survival, lifelines, and the official OpenAI
Python SDK. They are installed from their own distributions and remain subject
to their respective licenses and notices.

BrainVLM source and checkpoints are external inputs supplied by the user. They
are not covered by this repository's MIT license. Users are responsible for
obtaining them from an authorized distribution channel and complying with all
applicable upstream terms.

## UTSW-Glioma test case

The files under `test_data/utsw_glioma_bt0156/images/` are modified, selected
2D PNG derivatives from the UTSW-Glioma collection. The source data citation
is Reddy et al. (2026), *The University of Texas Southwestern Glioma MRI
dataset with molecular marker characterization and segmentations
(UTSW-Glioma)*, The Cancer Imaging Archive,
https://doi.org/10.7937/DFAE-1B86. The collection is available at
https://www.cancerimagingarchive.net/collection/utsw-glioma/ under CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/. See the case-level attribution
file for the full citation and modification notice.
