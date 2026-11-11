# Causal turn-anchored observations: prospective experiment

User proposes PreCoMem-like turn-level writing. Test one fixed writer strategy,
separately from the rejected relation-summary experiment. No new reader, answer
program, model sweep, or relation-summary context is introduced.

## Hypothesis and scope

Anchoring extraction to each incoming turn may retain specific observations and
attribute them more precisely than extracting a whole session at once. More
facts or links alone are not success; compare official final-answer F1.

For each of the 788 turns in conv-26/conv-30, supply that dated target, the eight
preceding raw turns, and up to eight relevant frozen facts from strictly earlier,
completed sessions (BM25 over the target and preceding context). Include their
original source turns. No future turn, question, gold answer or evidence label
enters the writer. Extract only NEW observations supported by the target, using
context to resolve references. Every accepted fact must cite the target; all
other citations must be visible causal sources. Empty extraction is valid.

This first test uses the completed-session fact cache as historical context;
newly extracted observations do not feed other writer calls. It is a causal,
parallelizable backfill strategy, not a reproduction of PreCoMem's full online
belief loop. Both granularity and writer context policy change, so attribute
any result to this writer strategy, not to isolated turn granularity alone.

Keep original raw turns/session metadata immutable. Store observations with the
target turn's position and date; run the original append-only exclusive-state
evolution rule. Events/preferences are never overwritten. Final retrieval uses
the SAME four-view source RRF and chronological raw-turn packing. Replace only
the fact view with the new facts, embedding them with the original local
Qwen3-Embedding-0.6B encoder, max length 2048, normalized float16 storage. Reuse
the original raw/query vectors exactly. The retrieved answer context contains
original turns, not the failed relation summaries. Its budget remains 3000
cl100k tokens. Reader: same gpt-6-luna low/768 ANSWER+COMPACT.

## Finite run and decision

All new generation is gpt-6-luna + Flex. Writer: low reasoning, max 2048 output
tokens, batches of eight independent causal targets. Preserve per-turn responses,
source rejections, incompletes, usage and request hashes; no automatic repair,
cap doubling, Standard fallback, or ambiguous request retry. Stop new batches
on transport failure or an unconfirmed Flex response. An incomplete response
adds no facts, retaining its original raw turn; stop if over 5% of all targets
are incomplete. The whole development build plus answer planning limit is
$1.50 using conservative byte-length cost proxies, not a provider invoice cap.

First verify causal source boundaries and target citation checks, unchanged raw
data, and exact original 80-context reproduction. Encode only new facts on one
free GPU; this is the existing 0.6B embedding model, not another local reader.
Then run the existing fixed 80 development questions once, with cached original
Luna low controls at zero new cost. Compare paired F1 and source recall, also
against the strongest cached reader (Luna medium 65.9598 F1). Expand once to
the separate 160 questions only if F1 exceeds that strongest reference by at
least 3 points and at most 5% answers are incomplete. Otherwise stop the fixed
writer strategy without a window/top-k/prompt grid. This reused development
result alone cannot establish a full-benchmark or SOTA breakthrough.

Status: written before writer preparation, new calls, embeddings or answers.

## Transport amendment, before further calls

The first eight-way Flex run terminates after 62 completed targets (86 accepted
facts, no incomplete response or source rejection), at estimated $0.01204604.
Two requests receive explicit HTTP 429 `flex_unavailable` rejections. Both are
recorded in `writer_failures.json`; all successful outputs are checkpointed.
[OpenAI's Flex guide](https://developers.openai.com/api/docs/guides/flex-processing)
states that this resource-unavailable error is not charged. No ambiguous timeout
or lost response is retried.

The user then explicitly authorizes Standard. Subsequent uncached calls request
`service_tier=default`, which the [official Responses reference](https://developers.openai.com/api/reference/resources/responses/methods/create)
defines as standard pricing/performance. Retain Luna and all semantic request
bodies, prompts, targets and cohort unchanged. Reuse the 62 successful outputs;
the 726 remaining targets include the two known unprocessed rejections. No
automatic Standard fallback is added: selection is explicit. New completions
record actual served tier, and per-tier cache hits remain zero-cost reuse.

Amend the conservative planning limit to $2.50 for the development build plus
answers, using Standard rates for remaining reservations; the initial $1.50
limit described Flex. Store previous and current implementation hashes and this
transport authorization in the run config. Availability recovery does not alter
the experimental hypothesis or enable an algorithm/prompt sweep.

## Completed result and decision

All 788 targets finish: 62 Flex and 726 Standard, 934 accepted observations,
no incomplete response or source-ID rejection. The old two memories contain
490 facts. Source-turn coverage rises 491 -> 649 of 788, but exclusive-state
version links remain zero in both new memories. Thus this is evidence about the
fixed writer strategy, not a demonstrated improvement to belief evolution.

Only new fact vectors are encoded. Old raw/query vectors are byte-equivalent;
all 80 old contexts reproduce exactly, and all new contexts fit 3000 tokens.
The 0.6B encoder's peak PyTorch allocated/reserved memory is 1.381/1.486 GiB;
its process exits after embedding. No local reader is loaded.

The frozen Luna low reader completes 80/80 via Standard with no cap. Official
F1 **65.4549 -> 63.5546**, delta **-1.9003 points**, question-bootstrap interval
**[-6.5300, +2.3629]**, 12 wins / 17 losses / 51 ties. Strongest cached Luna
medium is 65.9598; the candidate trails it by 2.4052 points. Source recall falls
**92.5855 -> 91.1538%**. The predefined expansion gate fails. Stop this writer
strategy: no 160/full expansion and no window/history-top-k/prompt grid.

Estimated new cost: writer **$0.35758886**, reader **$0.03262988**, total
**$0.39021874**, 868 new processed calls. The two explicit Flex rejections are
not included as processed calls. All reference answers are reused. These are
usage-based estimates, not a provider invoice or account balance.

Bounded after-generation diagnostic takes only the two largest losses and
largest gain. The birthday answer changes `10 years ago` to a longer phrase
with `Ten` and an extra source date, losing official lexical F1 despite source
recall remaining 1.0. A pottery-list answer drops bowls with source recall
unchanged at 2/3. A counterfactual-counseling answer improves from abstention to
`Likely no`, with recall 1.0 in both. These cases neither establish prevalence
nor separate source selection, contextual interference and answer variability.
No post-hoc answer normalization, semantic regrading or targeted repair is used.

The result rejects this particular turn-anchored backfill strategy, not all
turn-level or topic-level memory systems. Newly written beliefs did not feed
other writer requests, so this is not a negative reproduction of PreCoMem's
online confidence lifecycle. The broader metric-breakthrough goal remains unmet.

Artifacts: [official comparison](../../reports/turn_dev.json), ignored run
`runs/phase3/turn_dev/` with all requests, facts, predictions and the bounded
diagnostic. After execution, `prepare` metadata is aligned with the authorized
Standard tier/rates/limit; semantic request construction is unchanged. The exact
executed source/config are preserved in `executed_turn_study.py` and
`executed_config.json`, with execution hashes retained separately from current
metadata-construction hashes. No predictions are regenerated for this correction.
