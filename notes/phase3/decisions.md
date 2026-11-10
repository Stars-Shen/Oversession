# Research decisions: avoid repeating low-value experiments

All scores below use the official local LoCoMo F1 functions, not the accuracy
numbers quoted by many memory papers. Different reader/model cohorts must not be
combined. Earlier completed v2 benchmark results remain separate and reproducible.

## Continue from the useful signal

With frozen memories and the same source context, native DeepSeek thinking gives
59.68 → 65.40 F1 on the 160-question separate-conversation validation cohort.
Among the completed full-run pairs, excluding development and validation leaves
603 pairs at 60.04 → 63.94. This is an inference-compute gain; it does not establish
a new memory algorithm. Single-hop declines. The paid full run remains incomplete
after HTTP 402, so neither number is a completed 1540-question result.

## Stop these directions on present evidence

| Fixed hypothesis | Decision-relevant result | Decision |
|---|---|---|
| Weighted coverage/source bundles | DeepSeek development 64.16 → 60.81 | Stop packing-weight search |
| Query-local event table | DeepSeek development 64.16 → 63.23 | Stop uniform table compilation |
| Question-blind semantic graph + PPR | DeepSeek development 64.16 → 60.74 | Stop graph-growth tuning |
| Local cross-encoder source ranking | Development gain; separate validation 59.68 → 56.18 | Reject after failed transfer |
| Context-aware decoding, alpha 0.5 | Local Qwen development 58.40 → 59.25 | Stop alpha grid |
| Deterministic relative-date annotations | Same local cohort 58.40 → 59.81 | Keep modest result; no grammar expansion |
| Source-bound structured reading + one constrained repair | Same local cohort 58.40 → 58.51; all final IDs valid | Stop prompt/grammar variants |
| Bayesian change-point filtering on existing exclusive-state tracks | Only two usable adjacent cross-session observations in the 24-question LME development audit | Insufficient observation process; do not fit a decorative posterior |

The implication is not that these mechanisms never work. Their assumptions or
costs do not currently justify more experiments in this setup. A legal source ID
is a valid address, not an entailment certificate; a sharp retrieval distribution
is not calibrated truth; shared unitary rotations cannot change fixed-vector dot
products. Mathematical machinery must change an observable decision to be useful.

## Value-layer result and selected confirmation

The canonical-value study reuses teacher facts that already resolve names, times
and conversational references, instead of requiring the local reader to resolve
them again from raw turns. This tests the representation passed to the reader,
with the same RRF source priority and token budget. It also changes packing density,
so its interpretation must include both effects. Uniform replacement fails:
58.40 → 56.93 on the frozen 80-question development cohort. Temporal F1 nevertheless
improves by 12.89 points. One post-hoc selection uses the already fixed calendar
question rule to choose canonical values only for time questions, giving 62.52
overall (+4.12) without new generation calls. Freeze this rule for the existing
160-question separate-conversation comparison. Its completed result is
51.58 → 53.05 (+1.47), with a paired question-bootstrap interval [−1.54, +4.50]
points. The larger development gain did not sufficiently transfer. Stop this
rule without tuning or full-run expansion. Details: `canonical_values.md`.

Further cross-field reading includes static-analysis abstract interpretation
(`abstract_interpretation.md`). It supplies a useful precision/soundness distinction,
not a formal guarantee for extracted language-model facts or another experiment
to launch automatically. All local studies listed here have finished; their
baseline caches and original ledgers are preserved.

This file records decisions in the active authorized research task. It does not
instruct a future session to resume experiments or launch jobs automatically.

## Later capacity and learning checks

The [broader strong-memory review](strong_memory_followup.md) adds Hindsight,
SimpleMem, Memora and complementary curation/consolidation. Their high judge
accuracies must not be compared directly with official F1. A
[category-mixture audit](category_mix.md) further shows why the interrupted
DeepSeek cohort cannot establish a full-micro-F1 breakthrough.

Qwen3-4B native thinking improves its own 80-question development control
52.44 → 58.25, but the existing 30B plain reader is 58.40. Reject native-only
expansion; a large gain over a weaker baseline is not a new best result.

One [fixed low-rank reader adaptation](learned_reader_hypothesis.md), trained on
152 questions from conversation 0, gives +2.85 points on conversation 1's 81
questions. This misses the initial 3-point triage gate. One explicitly recorded
exception checks the unchanged final adapter on the existing 160-question
separate-conversation manifest: **46.25 → 45.37**, versus **51.58** for the
existing 30B reader. Reject this adapter; do not tune epochs/rank/temperature or
expand it to the full benchmark. All these jobs have finished. The broader goal
of a larger absolute breakthrough remains unmet.

A separate [30B source-supervised reader study](source_supervised_reader.md)
is active. It combines final-answer learning with training-only positive-source
attention coverage and token-normalized CE. QEMem, EviCover and explanation
regularization motivate the mechanism; E-mem supplies an adjacent intact-episode
alternative. No inference-time evidence-output schema is added. One final
adapter and one validation are scheduled, with no parameter search. Its real
maximum-length backward smoke passed; answer metrics remain pending.

A [task-risk transfer analysis](task_risk.md) distinguishes sequence-risk
training, MBR decoding and Bayes-optimal binary-set F-measure decisions. The
official multi-hop comma metric is not the latter objective. One cache-only
three-reader consensus test gives **65.57 → 67.07** on development, versus the
strongest existing DeepSeek reasoning reader, with no new model calls. Its
declared +3-point expansion gate fails; stop selector variants. A gold-only
pool ceiling is diagnostic, not achieved performance. The fixed 30B training
continues independently; no additional training or paid job is launched.
