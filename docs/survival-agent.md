# Survival Agent

The public Survival Agent contains the complete feature aggregation,
dimensionality reduction, random survival forest, and accelerated failure time
implementation used by the workflow. Model checkpoints and patient-level
training data are distributed separately.

## Feature contract

The aggregation functions consume a validated feature store with 4096
dimensional BrainVLM hidden features. The supported modes are `all_slices` and
`t1ce_keyframe`. Both modes exclude invalid slices; the keyframe mode also
requires a valid `keyframe_label` for every active T1CE slice.

The regularized autoencoder fits its scaler and model on training features and
applies the fitted transformation to the requested cohort. The random survival
forest uses observed-event indicators and positive follow-up times. The AFT
stage rank-normalizes forest risk against the training distribution before
estimating calibrated survival functions and median overall survival.

## Orchestrator contract

`MidlineOrchestrator` accepts `survival_workflow` as either a callable or an
object exposing `run`. It is called with `(case_input, diagnostic_result)` and
must return `midline_agent.schema.SurvivalResult`. Completed results require a
finite risk score, a positive median survival time, and a non-empty,
non-increasing survival curve over strictly increasing time points.

The orchestrator records an unconfigured workflow as unavailable, converts an
execution or validation error to a failed Survival step, and does not start the
Treatment Agent after a Survival failure.
