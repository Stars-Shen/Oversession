# Memory-R1: final-paper and pinned release assessment

## Evidence that changes the next decision

Yan et al., [Memory-R1: Enhancing Large Language Model Agents to Manage and
Utilize Memories via Reinforcement Learning](https://aclanthology.org/2026.acl-long.583/),
ACL 2026, pp.12805–12825, DOI 10.18653/v1/2026.acl-long.583. The final paper
adds GPT-5-trajectory behavior cloning as Memory-SFT. Table 1 reports SFT/GRPO
F1 of **42.81/45.02** for LLaMA-3.1-8B and **39.51/43.14** for Qwen2.5-7B.
LLaMA PPO is 41.05, below its SFT control. Thus the incremental GRPO gain over
SFT is 2.21–3.63 points, not the large gain over weaker untrained memory systems.
Our answer-only 4B update is not that trajectory-cloning control. The paper
describes EM reward; do not silently equate this with released F1 training.
Appendix D uses four 80GB H100s, eight for 14B; that is unlike our spare 24GB cards.
Targeted reading covered methods, SFT/results, implementation and limitations.

The locally downloaded final PDF has SHA256
`35f4f543501f2e2d1c8cc95503bae81a31b87c550e84c8977380b5ad3985a80e`
(3,697,363 bytes). Its Table 1 supersedes v1's ambiguous headline comparison
for future interpretation; existing v1 notes retain their explicit version scope.

## Current author code, rather than stale search snippets

The author repository is now populated even though search snippets still say
code is forthcoming. Read-only inspection is pinned to
[`yansikuan/memory-r1@f5ca10a45a08320b60575010c622fed6d3b3c686`](https://github.com/yansikuan/memory-r1/tree/f5ca10a45a08320b60575010c622fed6d3b3c686).
Its root license is Apache-2.0. A partial clone and selected files are stored only
under ignored `runs/phase3/paper_code/`; no author setup/training script ran and
no environment was installed. Existing model weights and baseline code are intact.

| Inspected release component | Observation and reuse consequence |
|---|---|
| [GRPO Qwen config](https://github.com/yansikuan/memory-r1/blob/f5ca10a45a08320b60575010c622fed6d3b3c686/answer_agent/scripts/train/configs/release/grpo_qwen25_7b.sh) | Five temperature-1 rollouts, prompt/response caps 4096/2048, batch 128, actor LR 5e-7, KL .001, up to 400 rollout steps. This entails far more sampling than our fixed three-epoch adapter. |
| [Selected reward entry point](https://github.com/yansikuan/memory-r1/blob/f5ca10a45a08320b60575010c622fed6d3b3c686/answer_agent/verl/trainer/main_grpo_f1_reward.py) | Instantiates `RewardManagerAnswer_F1`; older EM and mixed-memory classes in the same file are not the selected reward. |
| [Answer metrics](https://github.com/yansikuan/memory-r1/blob/f5ca10a45a08320b60575010c622fed6d3b3c686/answer_agent/evaluation/metrics.py) | Counter token F1 after case/punctuation/article normalization; no Porter stemming or category-specific comma matching. Reusing it would change our objective. |
| [Data guide](https://github.com/yansikuan/memory-r1/blob/f5ca10a45a08320b60575010c622fed6d3b3c686/answer_agent/data/README.md) | Public answer-agent parquet has 150 train and 81 validation rows, confirmed by existing PyArrow metadata; raw evidence maps and SFT trajectories are not bundled. This is a release observation, not a claim that the paper's 152-question split is invalid. |

The data schema is `data_source`, `prompt`, `ability`, `reward_model`, `extra_info`.
The selected files expose ground-truth answer supervision; no source map is in
their reward fields. Original release-derived data remains local and is not
copied into our benchmark or redistributed. The framework requires separate
verl environments and vLLM infrastructure; there is no pretrained checkpoint
in the inspected release to adopt immediately.

## Transfer assessment for this project

The reusable unit is the final-outcome training signal with a frozen memory
input, rather than a wholesale training stack. We already have an unchanged
official scorer and source manifests. Any later outcome-learning trial should
use our scorer on training questions, preserve the dialogue split, and account
for rollout/gradient compute. It must not silently change the metric, use
validation references in the objective, or advertise a direct reproduction.

The current 30B source-supervision run remains fixed. Its final 81-question
comparison will decide whether that mechanism transfers before any new training
is considered. The final paper does not justify launching RL merely because
our small answer-only adapter failed. No new training, inference, or paid call
was launched for this release inspection.
