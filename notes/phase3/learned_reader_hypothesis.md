# Learned memory use: a possible next mechanism, not a launched job

Yan et al., [Memory-R1, arXiv:2508.19828v1](https://arxiv.org/html/2508.19828v1),
2025-08-27. Targeted method, experiment and ablation reading distinguishes two
learned components: a memory manager chooses CRUD operations; an answer agent
filters retrieved entries and answers. Both use outcome rewards based on exact
match. Freezing the other component helps attribute an update's reward. The
answer agent retrieves 60 candidates before learned distillation. The paper's
split is by dialogue: 152 training questions, 81 validation, 1307 test, with
LLaMA-3.1-8B and Qwen2.5-7B backbones. Training uses four 80GB H100s, not our
three spare 24GB cards. Its v1 has inconsistent F1/BLEU labels between tables
and prose; no single ambiguous headline is used as a comparable result here.
No author code, weights or training service were imported.

Transfer hypothesis: keep the existing immutable memory and retrieval fixed,
and learn to use noisy but source-grounded context. This differs from asking an
untrained model to emit a structured evidence trace; that prompt trial failed.

Native local reasoning did not produce a material absolute gain: its 58.25 F1
remains below the existing 30B reader's 58.40. A simpler first test than full RL
is one small LoRA supervised reader adaptation.
Use only conversation 0's nonadversarial questions as training targets, conversation
1 for validation, and conversations 2–9 solely for subsequent evaluation. This
changes the scope: never describe the resulting trained model's score over all
1540 questions as an untouched test result. Preserve original checkpoints and
ledgers; write only a task-local adapter. Freeze memory, source context, prompt
and inference settings across the adapted/base comparison.

This is not a Memory-R1 reproduction or evidence that supervision alone will
work. It asks whether learned memory use transfers to new users before spending
on RL rollouts or a parameter grid. The current environment lacks PEFT/TRL;
the small standard low-rank update is implemented directly without installing
another dependency stack. This note alone does not instruct a future session to
resume training.

## One fixed adaptation, now active

The update follows [Hu et al., LoRA](https://arxiv.org/abs/2106.09685): frozen
linear weights plus a low-rank `alpha/r * B A` update. This is a standard equation
implementation, not copied author code. Only Qwen3-4B query/value projections
receive rank-8, alpha-16 updates. Train for three epochs with learning rate 1e-4,
gradient accumulation 8, clipping 1.0 and seed 20261111; use only the final adapter.
There is no validation-based checkpoint or hyperparameter selection.

Training sees 152 conversation-0 questions with gold final answers; validation
contains 81 conversation-1 inputs with no targets in the generation manifest.
The base reader, memory, source context, answer prompt, sampling parameters and
192-token output cap remain fixed. Twenty identical base-validation outputs are
reused; only 61 additional controls are generated. No test conversation contributes
a training target. Original weights are never written; the adapter is task-local.

Loss applies only to final-answer and EOS tokens. Computing only the last target
logits avoids allocating a full vocabulary tensor for the long prompt. A tiny
model regression verifies equality with the full masked causal loss. A second
checks exact initial logits and frozen original weights while adapter gradients
flow. The original torch environment suffices; no PEFT/TRL installation is used.

Require at least +3 paired F1 points and at most 5% incomplete outputs per condition
on the 81-question validation before checking the existing separate-conversation
cohort. A one-dialogue gain is not evidence of a full benchmark breakthrough.
Artifacts: `runs/phase3/reader_adapter/`, `local_adapter_baseline_validation/`,
`local_adapter_validation/`. Entry points: `src/reader_adapter.py` prepare, train,
prepare-evaluation and score; generation reuses `src/local_native.py` in off mode.
