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
adapter completes 51 training steps and the 81-question validation, with no
parameter search. Validation improves **58.62 -> 61.20 (+2.58)** with no capped
answers, below the original +3 triage gate. Preserve that false gate. Because
all three observed categories improve while inference has no validation cases,
one explicitly recorded exception checks the unchanged adapter on the existing
160-question, eight-conversation cohort. Require both +3 paired and +3
target-mixture sensitivity points before full expansion, with no further
exception or tuning for this candidate. That comparison completes **51.58 ->
55.70 (+4.12)**; target-mixture sensitivity is +4.29 points and neither reader
has capped outputs. Multi-hop nevertheless declines by 3.97 points; do not
claim a multi-source composition improvement. The same adapter now enters a
complete 1307-question paired comparison, reusing 160 answers per condition
and separately reporting the remaining 1147. Full-benchmark results are pending.

A [task-risk transfer analysis](task_risk.md) distinguishes sequence-risk
training, MBR decoding and Bayes-optimal binary-set F-measure decisions. The
official multi-hop comma metric is not the latter objective. One cache-only
three-reader consensus test gives **65.57 → 67.07** on development, versus the
strongest existing DeepSeek reasoning reader, with no new model calls. Its
declared +3-point expansion gate fails; stop selector variants. A gold-only
pool ceiling is diagnostic, not achieved performance. The fixed 30B training
continues independently; no additional training or paid job is launched.

The stronger same-context 160-question anchor is DeepSeek deliberate at 65.40,
versus the source adapter's 55.70. Its local-control gain is not a larger
absolute breakthrough. A fixed [full raw history study](full_history.md) tests
the information-budget hypothesis with the original unadapted 30B reader and
question-independent KV reuse. Expansion requires >=3 F1 points over the
strongest existing 80-question development reader (65.5748), not merely over
the weaker local control. No prompt or memory-profile grid is authorized by
that protocol. Existing 3K full evaluation continues on its original GPUs.

The full-history test completes at **50.97** versus local 3K retrieval's 58.40
and DeepSeek deliberate's 65.57, with no capped outputs. Reject this candidate
and stop expansion; do not sweep context order or length. Prefix reuse works
as a computation mechanism, but does not make the additional information useful
to the original reader. The next bounded feasibility assessment concerns the
same-size native-thinking checkpoint, not another summary/retrieval heuristic.

After the strongest-reference comparison, cancel the queued remaining 1147
source-adapter generations. Preserve existing outputs and checkpoint and allow
the original 1307-question control to finish as a reusable baseline. Pause the
orchestrator to prevent the adapted child from launching, then terminate that
paused parent once the control ends. This is an intentionally incomplete paired
study, not a completed full score. The next [fixed native 30B reader policy](native_30b_reader.md)
uses unchanged 3K evidence and a single model-card sampling policy, with a hard
4096-token budget and no prompt/temperature/budget grid. Generation is pending.

Original control completes **1307/1307 at 57.6954 F1**, retaining its one capped
answer; its paused orchestrator is terminated without launching the adapted
child. The old same-cohort Luna reference is 61.4174. A
[decision-theory interpretation](information_and_reader_risk.md) distinguishes
optimal information order from a fixed reader's realized loss, without treating
token budgets as measured Shannon capacity or inventing an information-loss
guarantee for a generated summary. Existing SF caches do not contain reasoning
text, so offline teacher-trajectory distillation is not silently assumed.

The user then requests a return to OpenAI with lower cost. Stop the two local
native-reader children and coordinator, preserve four outputs, and release the
owned approximately 124 GiB allocation. Do not continue local MoE execution
work. The [OpenAI economy protocol](openai_economy.md) uses original Luna through
Flex, preserving existing API cache and baseline code. The live pilot confirms
Flex service; one medium/4096 reader policy enters the fixed 80-question study,
reusing all 80 low/768 controls. Its conservative planning estimate is $0.153;
no automatic Standard fallback, writer rebuild, output-limit doubling or
ambiguous retry is permitted by this workflow. Full metrics are pending.

The fixed OpenAI trial completes **80/80**, at estimated new cost **$0.01745**.
Luna low/medium F1 is **65.4549 -> 65.9598**, versus strongest DeepSeek 65.5748;
none is capped. The +3-point expansion gate fails. Stop medium-reader expansion
and effort/budget sweeps. Retain the verified Flex transport and exact old-cache
reuse for subsequent OpenAI research. Saving cost is achieved; the larger final
accuracy breakthrough remains unproven.

The [Memory-R1 final paper and pinned release assessment](memory_r1_release.md)
replaces stale assumptions that author code is unavailable. It distinguishes
trajectory cloning, final-answer supervision and outcome learning, and records
the released reward/data/compute differences. Reuse the outcome signal only
after the current fixed candidate has an actual validation result; do not
install or launch the full public RL stack as an automatic response to a weak
SFT result. The ongoing training and evaluation queue remain unchanged.

The [Sol probe](sol_reader_probe.md) finishes **62.4512 F1**, below Luna medium's
65.9598, estimated cost $0.332399. Reject expansion. The user explicitly
clarifies **gpt-6-luna only, do not use Sol**. Lock the Flex request entry point
to Luna before network setup; archive the Sol script and retire its generation
command. No Sol job or queue remains. Subsequent research must use Luna and
preserve the cheaper-API preference. The broader goal remains unmet; this
rejection does not authorize a sibling-model sweep.

User directs work to representation/evolution/retrieval and asks about
PreCoMem-like turn-level writing. Keep that granularity question explicit;
session extraction has not been proved optimal. First complete one fixed
[incremental relation-memory study](relational_memory.md), holding extraction
and Luna low/768 use fixed. It creates 49 versions / 41 active relations / 8
local revisions, and packs complete original support groups inside 3K.
All 80 original controls reproduce exactly; the new run completes 80/80,
with 61 new answer calls and 19 exact-cache hits.

Result: **65.4549 -> 63.9217 F1 (-1.5332 points)**, 8 wins / 15 losses / 57 ties,
source recall **92.5855 -> 91.3034%**, estimated total new build+answer cost
**$0.05308243**. No capped answers. Expansion gate fails: stop the candidate,
no 160/full expansion and no lexical-gate/summary-count prompt grid. The bounded
diagnostic shows broad relation interference, and the largest gain is just
`Twice` -> `2 times`, not newly demonstrated counting ability. A turn writer
should be compared separately with precise source attribution; do not assume
more facts or finer segmentation alone is progress. The overall breakthrough
goal remains active and unmet.

The [fixed turn-anchored observation study](turn_writer.md) processes all 788
development turns with causal preceding context and frozen earlier-session facts.
Flex explicitly rejects two requests as `flex_unavailable` after 62 successes;
the process stops and checkpoints all successes. The user then authorizes
Standard. Resume only 726 outstanding targets via explicit `default` tier,
keeping Luna, prompts and experimental semantics unchanged. No ambiguous retry
or automatic paid-tier fallback is introduced.

All 788 writers and 80 readers complete without truncation; sources pass
membership/target checks. Fact count rises **490 -> 934**, source-turn coverage
**491 -> 649**, but exclusive-state version links remain zero. The original 80
contexts reproduce exactly, with unchanged raw/query vectors and the same 3K
packing/reader. Matched Luna F1 **65.4549 -> 63.5546 (-1.9003 points)**,
bootstrap interval **[-6.5300, +2.3629]**, 12 wins / 17 losses / 51 ties.
Source recall **92.5855 -> 91.1538%**. Estimated writer/reader costs
**$0.35758886 / $0.03262988**, total **$0.39021874**. Stop this fixed writer
strategy, no 160/full expansion or window/history-top-k/prompt grid. It is not
a negative reproduction of the full PreCoMem online belief loop. The broader
breakthrough remains unmet.

A separate [Late Chunking reading](contextual_embeddings.md) considers context
before embedding pooling, with explicit limitations: its v1 BEIR result uses
Jina mean pooling, while the current Qwen checkpoint uses last-token pooling.
No model is downloaded, code copied, pooling changed or new experiment launched
for that reading.
