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
160-question separate-conversation comparison; do not tune it after seeing those
answers. This development gain is not confirmation. Details: `canonical_values.md`.

This file records decisions in the active authorized research task. It does not
instruct a future session to resume experiments or launch jobs automatically.
