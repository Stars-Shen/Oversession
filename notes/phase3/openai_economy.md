# Return to OpenAI with a bounded Flex reader study

Prospective protocol, 2026-11-11, before the first new paid request. The user
explicitly requests returning to OpenAI and finding a cheaper approach.
Stop the two local native-reader processes and their coordinator; preserve the
four saved outputs, inputs, checkpoint and logs. This is a provider-switch
cancellation, not a completed native 80-question score or a model-quality failure.
No local generation is to be resumed by this protocol.

Use the existing `gpt-6-luna` model through an isolated opt-in Flex client.
Keep shared `src/api.py`, original memory ledgers, baseline configurations and
old API cache unchanged. Exact old request signatures are looked up before
making new requests. Completed old answers cost zero to reuse. A new cache
stores actual service tier, token usage and current-rate cost estimates.
No automatic Standard fallback, ambiguous network retry or output-cap doubling.
Only returned `service_tier=flex` establishes the cheaper served tier.

Official [Flex documentation](https://developers.openai.com/api/docs/guides/flex-processing)
and [pricing](https://developers.openai.com/api/docs/pricing) show 50% lower
token prices than Standard for this model. Luna's short-context Flex prices
per million tokens are input $0.05, cached read $0.005, 30-minute write $0.0625,
output $0.25. Input buckets are disjoint; reasoning is already included in
output tokens. Estimates use current rates, not historical invoices.
Prompt cache hits are observed through usage rather than assumed; do not pad
or expand evidence just to force caching. Batch is an alternative for delayed
offline work, not another job automatically submitted by this experiment.

To advance the original metric goal, test exactly one reader policy: Luna
`reasoning.effort=medium`, max output 4096, against the cached same-input
Luna low/768 control on the fixed existing 80 development questions. All 80
old low answers exist and score 65.4549. The strongest current same-question
reference is DeepSeek deliberate at 65.5748. Keep the exact 3K context, system
instructions and question text. No writer rebuild, additional retrieval or
model/effort/budget grid. Costs and any incomplete outputs remain visible.

First send one useful question from that manifest as the Flex connectivity
pilot; reuse its response for the remaining 79, never regenerate it as a
separate smoke test. Calculate the conservative public-rate planning cost
before expanding, targeting <=$0.20 for the complete 80-question trial. A
live transport error or unconfirmed Flex tier stops automatic expansion.
Score after all 80 outputs exist. Expand further only if >=3 F1 points above
the strongest cached reader and <=5% incomplete responses. Provider and
reader effort changes are explicitly recorded; no new memory-algorithm claim.

This records the active authorized task, not instructions to resume jobs in
another session without the user's authorization.

## Cost audit and live pilot

35,118 saved Responses contain 127,980,702 input and 15,665,643 output tokens;
1,708,534 input tokens are cache reads and 124,930,757 are cache writes. Repricing
these completed records at current Luna rates gives Standard $23.6004 versus
Flex/Batch $11.8002 for the same usage. This is not historical invoicing, and
does not cover missing/failed calls, embeddings or other judge endpoints.
`reports/openai_usage_audit.json` records scope and counts. The low read-hit
fraction does not support assuming a 90% caching discount on the whole workload.

The new useful pilot returns `service_tier=flex`, 3191 input tokens (3188 cache
writes), 107 output tokens including 95 reasoning tokens, estimated **$0.00022615**.
It is reused for the full trial. The fixed 80-question input planning estimate
is **$0.152877** even with all output limits used, using a conservative byte-count
input proxy. Actual usage will be reported separately; this is not an invoice
guarantee. Old Luna low F1 is 65.4549 and the strongest DeepSeek control is
65.5748; these are retained as separate quality anchors.

## Completed fixed trial

All 80 outputs complete without capping. Estimated new API cost is **$0.01745294**
for exactly 80 paid responses including the reused pilot once. The same returned
usage at Standard rates would cost twice as much. Existing low answers are
reused with zero new cost; these are public-rate estimates, not account invoices.

Official F1: low **65.4549**, medium **65.9598**, strongest DeepSeek **65.5748**.
Medium gains only **+0.5049** over low and **+0.3850** over the strongest control;
the latter's paired interval is [−4.0641, +4.9393] points. The +3-point expansion
gate is false. Stop this candidate without effort/budget/prompt/model sweeps or
160-question expansion. The cost transport is useful; it is not the requested
larger accuracy breakthrough. `reports/openai_flex_reasoning_dev.json` records
all metrics, actual service tiers, paid-call counts and cost scope.

Two targeted no-network checks pass: legacy cache reuse sends no paid request,
and cost arithmetic partitions input buckets without charging reasoning twice.
The original shared API and baseline paths remain unchanged. The new client
is opt-in via `src/openai_flex.py`; research calls in this workflow use it.

The user's subsequent model restriction is explicit: **gpt-6-luna only**.
The request method rejects other models before opening a session. A targeted
no-network check confirms the restriction. Historical alternative-model prices
are accounting records and do not authorize new calls.
