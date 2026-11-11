# Incremental cross-session relations: prospective protocol

The user redirects research to representation, evolution and retrieval. Freeze
extraction and use: original session-level fact caches, original captions,
original Luna low/768 ANSWER+COMPACT reader, and 3000 cl100k tokens. All new
generation uses gpt-6-luna through Flex; no local reader or Sol jobs.

## Hypothesis and mechanism

The two development memories have 19 sessions each, 252/238 facts, and zero
exclusive-state version links. This does not prove retrieval failure, but exposes
a missing operation: binding relations whose premises arrive in different
sessions. Test whether question-independent materialized relations improve final
answers while their complete original witnesses still fit the same context.

1. Representation: keep raw turns and extracted facts immutable. Store precise
   cross-session relation cards with subject/relation/scope keys and support
   groups. Within each group all fact roots are jointly required (AND); groups
   are alternatives (OR). A card is an LLM derivation, not a verified theorem.
2. Evolution: process sessions in order. Each new group uses 2–4 known facts,
   spans at least two sessions, and includes a fact from the incoming session.
   Append a version only when a same-key active card needs a concrete refinement
   or correction. Retain the old card and immutable facts. No unrelated global
   confidence updates, inferred plan completion, or trait summaries.
3. Retrieval: rank active relation text with stemmed BM25. Require a non-entity
   content-word match and at least one witness in the original 3K selection.
   Select at most two cards with distinct keys. Choose one complete support
   group per card by mean original source-RRF score, then deterministic ID tie.
   Pack the card and ALL its original turns atomically. Fill remaining budget
   with the unchanged original source-RRF ranking in chronological display.
   Recompute actual serialized tokens on additions. No partial provenance.
   No eligible/fitting card, or disabled feature, returns the exact old result.

This transfers focused contextual evolution from A-Mem/RGMem and AND/OR
bookkeeping from positive database provenance. It is different from the rejected
broad semantic-node random walk (60.74 vs 64.16 F1): no graph diffusion, no
full vote per child, and derived relations are supplied with their witnesses.
It is also different from old canonical values: the new object is a relation
between existing facts, not another serialization of each independent fact.
See [primary readings and limitations](research.md#database-provenance).
The prototype rereads the fact prefix; no sublinear online-cost claim is made.

## Fixed experiment and decision

Build only conv-26 and conv-30 first. The builder receives an allowlisted fact
prefix and prior active cards, never questions, gold answers or evidence labels.
At most eight cards per session, 4096 output tokens, low reasoning. Save request
hashes, prefix checkpoints, rejections, usage and new cost. Stop if a new request
is not confirmed Flex. No automatic retry, cap doubling or Standard fallback.

Use the existing fixed 80 development question IDs. Before new reader calls,
verify exact reproduction of all original retrieved contexts from frozen
embeddings, and structural tests for chronological support, local revisions,
immutable input, complete support groups and context budget. These are integrity
checks, not semantic entailment tests. Then generate once using the frozen
reader. Reuse cached original Luna low controls at zero new cost.

Planning limit: $0.50 across build and development generation, using the current
Flex rate estimate; not an invoice or hard provider billing guarantee. Compare
paired official F1 to Luna low (65.4549), and also to the strongest cached reader
(65.9598). Report source recall, coverage, cost and capped outputs. Expand once
to the separate 160 cohort only if development exceeds the strongest cached
reader by at least 3 F1 points and at most 5% outputs are incomplete. Otherwise
stop this fixed candidate; no card-count, query-gate or writer-prompt grid.
Development improvement is not a SOTA or full-benchmark claim.

## Status

Completed once; rejected, with no 160-question expansion or prompt/gate grid.

- Build: 36 Luna Flex calls, 49 total versions, 41 active cards, 8 revisions;
  estimated new cost $0.04061184. Original memory files remain unchanged.
- Integrity: 80/80 original contexts reproduced exactly from frozen embeddings;
  61 new contexts contain relations, all with complete selected support groups;
  maximum context 3000 tokens. Four relation integrity tests and four Flex
  transport/accounting tests pass. These do not verify textual entailment.
- Generation: all 80 completed, 61 new calls and 19 exact old-cache hits. Same
  Luna low/768 reader and original answer instructions. Build plus generation:
  97 new paid calls, estimated $0.05308243, not a provider invoice.
- Official F1: **65.4549 -> 63.9217**, delta **-1.5332 points**,
  paired question-bootstrap interval **[-5.6278, +2.2545]**, 8 wins / 15 losses /
  57 ties. Stronger cached Luna medium is 65.9598; delta -2.0381 points.
- Source recall: **92.5855 -> 91.3034%**; no additional gold source recovered,
  one lost. The +3-point expansion gate is false.

Bounded post-generation diagnostic inspects only two largest losses and the
largest gain, not their population frequency. The largest loss answers common
painting subjects as nature/flowers instead of sunsets while gold-source recall
stays 0.5. Both retrieved cards are about Melanie's painting process/creative
outlets, not the intersection of two people's painted subjects. The next loss
switches a recent painting answer to a horse, with recall still 1.0. The largest
gain is merely `Twice` -> `2 times` under the official lexical metric; it is not
evidence of improved counting. Shared words or years plus a baseline witness
are insufficient to establish relation relevance. Do not repair this result by
scoring a post-hoc gate or summary-count sweep. Adding this class of derived
text to the answer context is not justified by the result. These cases do not
isolate the generated text from changes to the other packed source turns.

Separate extraction-source diagnostic on reused development: 111/118 annotated
source references are covered by existing facts; five references are absent
from both fact roots and the baseline context, affecting four questions. Lack
of a source pointer does not prove that its proposition was omitted elsewhere,
and pointer coverage does not prove an extracted fact preserved every detail.
This bounds the simple missing-turn explanation without dismissing the user's
[turn-granularity candidate](turn_granularity.md).

Artifacts: [completed official comparison](../../reports/relational_dev.json); ignored run
`runs/phase3/relational_dev/` contains ledgers, request hashes, inputs, predictions,
scored rows and the bounded diagnostic. Entrypoint: `src/relational_study.py`
commands `prepare`, `build`, `contexts`, `generate`, `score`, in that order.
The broader final-metric breakthrough remains unproven.
