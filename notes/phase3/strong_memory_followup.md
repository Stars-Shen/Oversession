# Strong-memory follow-up: metrics, assumptions and useful transfer

Reading date: 2026-11-11. Targeted primary-paper methods, result tables and
evaluation sections were read; this is not a reproduction of all four systems.
No paid API calls, author training scripts or database services were launched.

## Hindsight

Latimer et al., [ACL 2026 System Demonstrations, pp. 275–285](https://aclanthology.org/2026.acl-demo.27/),
with the [longer architecture preprint](https://arxiv.org/html/2512.12818v1).
Four networks distinguish facts, experiences, entity observations and opinions.
Recall combines semantic, lexical, graph and temporal channels with RRF and a
cross-encoder; reflect reasons over the retrieved memories. The demo reports
83.6% LongMemEval and 83.2% LoCoMo accuracy using GPT-OSS-20B. These are judge
accuracies, not our official LoCoMo F1; the longer v1 uses GPT-OSS-120B judging
and still contains retrieval-budget placeholders. The demo summarizes that
evaluation rather than providing a matched 3000-token reproduction protocol.

The MIT [official repository](https://github.com/vectorize-io/hindsight) now has
additional decision models and orchestration. Current main must not be treated
as the exact paper snapshot. Its 0.8B decision model requires newer dependencies
and is not installed in this study. Transfer: separate evidence from belief and
measure native reader reasoning before adding more graph channels.

## SimpleMem

Liu et al., [arXiv:2601.02553v1](https://arxiv.org/html/2601.02553v1), 2026-01-05.
Structured compression, consolidation and query-dependent retrieval aim to reduce
context cost. Reported GPT-4.1-mini average F1 is 43.24 with roughly 531 tokens;
its 26.4% improvement is relative, not 26.4 percentage points. The v1 describes
1986 questions while naming four task types, so it does not establish equality
with our 1540 category-1–4 evaluation. “Semantic lossless” is a design objective,
not a proof that an LLM summary preserves every query-relevant detail. Our failed
uniform canonical-value trial is a direct warning against assuming that property.

The MIT [official evaluator](https://github.com/aiming-lab/SimpleMem/blob/main/test_locomo10.py)
was inspected without importing it. The retrieved source uses set-token F1 with
simple punctuation/lowercase normalization, not our official Porter/category
scorer; it also has a separate category-5 multiple-choice path. This reinforces
the metric mismatch, without claiming to identify the exact script behind the
paper's tables. Download SHA256:
`d5fef37a21700e3c721b1556e70bef2cf53e35f8c8baca24f175a9755af02978`.
GitHub API rate limiting prevented pinning current main; no code was copied.

## Memora

Xia et al., [arXiv:2602.03315v1](https://arxiv.org/html/2602.03315v1).
Primary abstractions and cue anchors index rich values, while episodic context
preserves narrative grounding. Refine/Expand/Stop actions support active retrieval.
Main prompt-policy results are distinct from GRPO experiments with separate
10%/10%/80% partitions. In Table 1, semantic → policy retrieval changes LoCoMo
judge accuracy 84.9 → 86.3, but F1 only 55.2 → 55.3; full-context F1 is 56.5.
LongMemEval accuracy improves 83.8 → 87.4 while reported context increases
2.1k → 2.9k. This is useful evidence for active retrieval, not a free or uniformly
better transformation at our fixed budget.

The MIT [official README](https://github.com/microsoft/Memora) was inspected.
No services or training were installed. Its raw-episode option is relevant to
reference resolution, but our existing cross-encoder already packed three-turn
windows and failed transfer. Do not repackage that failed trial as a new idea.
Its framework-containment analysis is not a guarantee of better answer accuracy.

## Complementary curation and consolidation

Gupta et al., [arXiv:2610.39790v1](https://arxiv.org/html/2610.39790v1), 2026-10-29.
Episodic records and relational tuples provide complementary retrieval; offline
clustering merges or deletes entries. GPT-4.1-mini LoCoMo judge accuracy 90.2
is below its full-context 91.3, while using fewer tokens; corresponding F1 is
35.4 versus 36.8. The consolidated variant falls to 89.2 judge accuracy and
34.8 F1. Its 85.5 LongMemEval accuracy is interesting, but does not justify
copying deletion rules that could erase historical evidence. Transfer: preserve
complementary access paths and measure their downstream utility, retaining raw
ledgers. No author code was reused.

## Decision for this project

High headline judge accuracy does not by itself identify a stronger official-F1
method. Several mechanisms overlap with failed local studies. The evidence now
supports testing native reasoning on a practical local backbone rather than
another untrained graph or summarization variant. The matched experiment is
specified in `local_native_capacity.md`; it must beat both its own off control
and the existing local 30B reader before expansion. It is a reader-compute test,
not a reproduction or new memory-algorithm claim.
