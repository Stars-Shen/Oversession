# Native reasoning as a local capacity intervention

The broader paper review is in `strong_memory_followup.md`. In the existing local
80-question development cache, 70 questions include all annotated source IDs;
they still account for 92.05% of the total F1 loss. In the separate 160-question
cohort, 130 have complete annotated-source coverage and account for 73.96% of
the loss. `scripts/audit_reader_bottleneck.py` records the exact fractions.
This does not prove the context is semantically sufficient: annotations can omit
antecedents, and official F1 also penalizes wording. It does make another generic
retrieval-channel expansion a weak first choice.

The next fixed hypothesis is that a locally runnable model with trained native
reasoning can exploit the existing memory better than a larger nonthinking reader.
The prior structured-working-state trial did not train that capability and failed.
This is a reading/inference-compute experiment, not a new memory algorithm.

## Frozen experiment

Use official [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), Apache-2.0,
revision `1cfa9a7208912126459214e8b04321603b3df60c`. The existing read-only
Transformers 4.51.3 environment supports its dense architecture directly.
Approximately 8 GB of weights are downloaded to the task-local ignored directory;
SHA256 is checked against Hugging Face LFS metadata during streaming. No installed
library or previous checkpoint is changed.

Copy exactly the answer-free 80-question v2 source-packed manifest used by the
local 30B baseline. Keep its system instructions, user prompt, frozen Luna writer,
source IDs and 3000-cl100k-token evidence unchanged. Toggle only the official
`enable_thinking` template argument. Use identical sampling parameters in both
modes: temperature 0.6, top-p 0.95, top-k 20, min-p 0, deterministic per-question
seeds. The official card discourages greedy thinking; common sampling parameters
avoid confounding the toggle with different temperatures. This is one seed per
question, not a multi-seed reproducibility claim.

Completion limits are 192 off and 4096 on. Longer inference is part of the
intervention, and usage is reported. Parse the final answer after the last native
`</think>` token; unfinished reasoning or missing final text becomes an abstention
and remains in scoring. Preserve reasoning separately. The parser isolation and
truncation behavior have targeted tests. Use the unchanged official local LoCoMo
scorer only after generation; no category or gold enters the model.

Before generation, require all of: at least 3 paired F1 points on versus off,
at least 3 points above the existing **30B plain reader** on the same questions,
and at most 5% incomplete outputs in each mode. Only then expand to separate
conversations. This prevents claiming success by choosing a weaker comparison
baseline. No prompt, temperature or completion-budget grid is planned.

Artifacts: `runs/phase3/local_native_capacity_dev/`. Preparation/scoring use
`.venv/bin/python src/local_native.py prepare|score`. Generation uses the original
torch Python and `src/local_native.py generate`. The off control runs once on
physical GPU 1; on questions are partitioned into two disjoint modulo shards
on physical GPUs 2 and 3, bound by UUID. Each process writes a distinct JSONL,
and scoring requires the exact question-by-mode set without duplicates. Seeds
depend on question ID rather than execution order. The model weights and complete
generation configuration are shared across all workers.

The SiliconFlow queue remains stopped after insufficient balance. These local
outputs cannot replace the missing exact-model DeepSeek full-evaluation rows.

## Result: paired gain, no absolute breakthrough

All 80 pairs completed. Off F1 is **52.4384**, on **58.2469** (+5.8086 points),
27 wins / 19 losses / 34 ties. The paired question-bootstrap interval is
[−0.6138, +11.9315] points. Each mode has one capped output; all are retained.
On uses 56251 completion tokens, including 51286 reasoning tokens; off uses 1555.
Temporal improves substantially, inference declines, and single-hop is nearly
unchanged. This is one sampling seed, not a multi-seed robustness claim.

The existing 30B plain reader scores **58.4042** on exactly these questions.
The new on condition is **0.1573 points lower**, failing the predefined absolute
gain criterion. Stop native-only expansion; the +5.81 gain over a weaker control
must not be presented as a new best score. `reports/local_native_capacity_dev.json`
records the complete comparison. This result motivates the separately specified
learned-reader adaptation; it does not authorize a temperature or budget grid.
