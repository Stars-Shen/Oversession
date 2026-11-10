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

If native local reasoning does not produce a material absolute gain, a simpler
first test than full RL would be one small LoRA supervised reader adaptation.
Use only conversation 0's nonadversarial questions as training targets, conversation
1 for validation, and conversations 2–9 solely for subsequent evaluation. This
changes the scope: never describe the resulting trained model's score over all
1540 questions as an untouched test result. Preserve original checkpoints and
ledgers; write only a task-local adapter. Freeze memory, source context, prompt
and inference settings across the adapted/base comparison.

This is not a Memory-R1 reproduction or evidence that supervision alone will
work. It asks whether learned memory use transfers to new users before spending
on RL rollouts or a parameter grid. The current environment lacks PEFT/TRL;
no dependency stack or training job has been installed or launched. Evaluate the
running native-thinking comparison first. This note alone does not instruct a
future session to resume training.
