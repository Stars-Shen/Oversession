# One-pass global memory lookup, prospective protocol

Prior goal turn is progress: the completed turn-writer comparison changes the
next decision (more observations alone did not improve final F1). Stop that
strategy, preserve its receipts, and return to the original frozen v2 memory.

## Primary readings and transfer

Lee et al., [ReadAgent, arXiv:2402.09727v1](https://arxiv.org/html/2402.09727v1),
15 February 2024. Read methods 3.1–3.4, task results, compression analysis and
the distracting/incorrect-lookup cases in appendix C. Ordered gists allow an LLM
to select raw pages jointly; parallel lookup uses one selection call. More
lookups can add distractors. PaLM 2-L QuALITY accuracy is 85.83 for full text and
86.63 for parallel lookup up to five pages. The GPT-3.5 parallel variant is
69.65, below full text's 73.30. Its result is not a general guarantee across
models, nor our official LoCoMo F1. Gist length also limits database scale.

Sun et al., [RankGPT, arXiv:2304.09542v1](https://arxiv.org/html/2304.09542v1),
19 April 2023. Read methods, passage-ranking tables, windowing, distillation,
cost and conclusion. Identifier-based listwise permutation avoids needing
relevance log probabilities. Its results concern nDCG on retrieval benchmarks;
it reranks a first-stage candidate set, and its sliding windows reread passages.
Those are not evidence of improved multi-source dialogue answering. No author
code is copied and neither paper is reproduced here.

Transfer: a single Luna call can inspect the global memory index and choose a
joint evidence set; source lookup supplies raw values to the frozen reader.
This differs from the failed 0.6B pointwise cross-ranker, node diffusion, added
relation summaries, canonical fact values, and cached answer-candidate selector.
It also differs from evidence-first answering: no answer or generated observation
from the selector enters final reader context. The earlier EviCover reading
warns against expensive coverage-only reflection, so use ONE fixed selection
pass, not an iterative selector/prompt/window sweep.

## Fixed mechanism

Question-blind index: original facts with compact aliases, subject/slot, dates
and text, plus original raw turns not cited by any fact (residuals). An allowlist
excludes QA and labels. Alias maps retain all original source roots. One query-time
Luna low/1536 call returns only selected aliases. It should match requested people,
relation, time and all list/count components, rather than just shared vocabulary.

Resolve the WHOLE selected set to original sources. Include the immediately
preceding turn of each root within its session for local dialogue context.
Render original dated turns chronologically. Do not append generated facts,
selection explanations, gists or answer guesses. Do not fill unused context
with lower-ranked distractors. Maximum final context: 3000 cl100k tokens.

Unknown aliases, empty selection, incomplete selector output, or a whole set
exceeding 3K returns the EXACT old context; no partial proof, truncation, repair,
automatic retry or hidden additional lookup. This structural policy does not
prove semantic entailment or actual evidence sufficiency.

Only gpt-6-luna + explicitly authorized Standard; preserve actual served tier,
usage, request hashes and per-stage cost. Frozen extraction/evolution, same final
Luna low/768 reader system, same 80 development question IDs and official scorer.
Old Luna low/medium references are cached. No local reader or GPU is needed.
The selector reads the global index in addition to the final reader's context;
this spends MORE inference compute on retrieval, not a matched-total-token test.
Conservative development planning limit: $1.00, estimates not provider invoices.

Before calls: verify index allowlisting, root/context closure, all-or-baseline
fallback and chronological rendering. Then one fixed 80-question run. Expand
once to the separate 160 only if F1 exceeds the strongest cached reader (65.9598)
by at least 3 points and at most 5% answers are incomplete. Otherwise stop this
candidate without a selection-size, padding, neighbor, effort or prompt grid.
Reused development alone cannot establish a SOTA/full-benchmark breakthrough.

Status: recorded before preparing selector inputs or generating new responses.
