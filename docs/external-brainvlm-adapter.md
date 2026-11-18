# External BrainVLM Adapter

The public package does not vendor BrainVLM source, model weights,
patient data, or generated artifacts. Obtain a compatible external checkout
and checkpoint through the project release channel, then pass both paths to
`midline_agent.diagnostic.brainvlm_adapter.BrainVlmAdapter`.

The constructor contract is:

```text
BrainVlmAdapter(source_root, checkpoint, "module_name:load_model")
```

The factory is imported from `source_root`, receives
`checkpoint=<absolute path>`, and must return an object exposing:

```text
predict(*, task, prompt, image, metadata) -> mapping
```

The mapping must contain a task label. Tumor tasks must also return a finite
five-value `logits` vector. The combined tumor/WHO task must return constrained
text such as `<tumor_2> <grade_3>` and a canonical label such as
`tumor_2|grade_3`. Missing or invalid labels are errors; no default labels are
invented.

The shallow public dataset modules define the three supported task names,
label vocabularies, prompt version, deterministic MRI preprocessing metadata,
and input/output SHA-256 checksums. `predict_who_grade` remains available as a
constrained bridge into the Diagnostic workflow.

External imports are restricted to the supplied source root. The loader
temporarily isolates module names and the import path, validates dependency
origins, restores process import state, and rejects path or symlink escapes.
The external runner remains responsible for its own architecture and license.

`scripts/train_brainvlm.sh` is only a portable launcher. It requires explicit
external source, checkpoint, training entrypoint, and configuration paths. No
training data, project-specific checkpoint, or external model implementation
is included here.
