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

## First validation and explicit triage exception

The fixed training finished in 863.6 seconds, with 2,949,120 trainable parameters,
8.91 GiB peak CUDA memory, and final adapter SHA256
`de076c407561d6a1e6b06172cc91fc5e9381f175083a93f44db2de10daa44507`.
All 81 validation pairs completed: **49.2796 → 52.1268 F1 (+2.8472 points)**,
22 wins / 20 losses / 39 ties. The question-bootstrap interval is
[−2.2324, +8.1234] points. The base has one capped output; the adapter has none.
The originally declared +3-point triage gate is **not passed** and stays false
in `reports/reader_adapter_validation.json`.

One explicit decision exception was made before inspecting any new-conversation
scores: because this is only one new user and falls 0.153 points short of an
arbitrary triage threshold, keep the exact final adapter and check the existing
160-question manifest across conversations 2–9 once. This follows the user's
preference for a decisive final-metric check over perfect intermediate gates;
it does not redefine +2.85 as success. No retraining, parameter change, sample
selection, or second exception is planned for this candidate. These conversations
were used in earlier research revisions; they are not an untouched final test.

Full evaluation requires at least +3 paired F1 points over the identical 4B base
and at least +3 above the existing 30B reader on these same 160 questions, with
at most 5% incomplete outputs per condition. Preparation/scoring:
`src/reader_adapter.py prepare-separate|score-separate`; generation uses
`src/local_native.py` with the `local_adapter_baseline_separate` and
`local_adapter_separate` runs. The adapter SHA256 is checked on load and recorded
in actual generation runtime metadata. The native-only 4B trial remains rejected.

## Separate-conversation result: reject this adapter

All 160 base/adapted pairs completed: **46.2491 → 45.3654 F1 (−0.8837 points)**,
30 wins / 39 losses / 91 ties, question-bootstrap interval [−5.3621, +3.7967].
Five base outputs hit the cap; the adapter has none. The existing 30B reader on
the same questions is **51.5752**, so the adapted reader is **6.2098 points lower**.
The final expansion gate is false; no full run or further tuning of this adapter
is justified. `reports/reader_adapter_separate.json` preserves the negative result.

The task effects differ: single-hop improves 61.028 → 65.940, while multi-hop
falls 56.316 → 48.606. This is not a universal improvement in memory use. A check
for unsupported appearances of the training speakers' names found none in either
condition, so literal name leakage does not explain the failure. Avoid asserting
that mechanism or launching a name-augmentation grid from this result alone.
The fixed supervised adaptation is distinct from Memory-R1 RL; its failure does
not establish that outcome-trained evidence selection cannot work.

## A useful failure diagnostic, not another training run

The adapter substantially shortens answers: multi-hop median completion length
falls **11 → 5** tokens, and its uncapped mean falls **16 → 6.6**. Single-hop
median falls 9 → 6.5. The associated task tradeoff is consistent with a brevity
bias, but does not establish that shortening caused every error.

The implemented loss averages token CE within each question, then averages
questions. With answer length T including one EOS, the EOS coefficient in the
dataset objective is E[1/T]. A globally token-normalized objective instead has
coefficient 1/E[T]; Jensen gives E[1/T] >= 1/E[T]. The 152 training targets have
mean T=7.9868, min 2, max 20, and coefficient ratio **1.3712**. These coefficients
are not measured gradient magnitudes or a proof of premature stopping; token
normalization also changes how much long answers influence learning.

This identifies a more concrete next hypothesis than another rank/epoch sweep:
learn complete evidence/list coverage, and distinguish it from learning short
answer style. Any future test should start from a competitive baseline and
preserve the fixed dialogue split. No changed loss or evidence-supervised model
has been trained in this study. Diagnostic target lengths are saved at
`runs/phase3/reader_adapter/length_bias_diagnostic.json`.
