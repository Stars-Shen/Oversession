# Canonical facts as the memory value layer

The source-packed v2 reader already retrieves using both original turns and
teacher-extracted facts. It then discards the fact text and passes original turns
as values. Two inspected, answer-free cases suggest that the local reader can
lose information already resolved by the writer: relative-month normalization
and separating a person's preference from an automatic image caption.

## Related primary work

Xu, Shi and Choi, **RECOMP**, ICLR 2024
([paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/bda88ed2892f5e61c9a9bf215c566913-Paper-Conference.pdf)),
sections 3.2 and 5, trains compression using downstream model utility. Its
abstractive compressor distills teacher summaries and filters them with task
signals; it is not generic summarization. Table 2 is a useful caution: on NQ,
five uncompressed documents achieve 48.28 F1 and the learned abstractive
compressor 45.47, despite substantially fewer tokens. Compression is not an
automatic accuracy improvement. Our frozen, question-blind facts require neither
new teacher calls nor downstream-label training, so this study is not a RECOMP
reproduction.

Chen et al., **Dense X Retrieval**, EMNLP 2024
([paper](https://aclanthology.org/2024.emnlp-main.845/)),
sections 2 and 6, uses self-contained atomic propositions and evaluates prompt
granularity under fixed token limits. It motivates testing the information
actually presented to the reader, beyond whether a source was retrieved.
Its English-Wikipedia experiments do not establish temporal conversational
memory performance. Our fact writer and source-priority ranking remain frozen;
no author implementation is copied and no propositionizer is trained.

## One fixed intervention

Obtain exactly the existing four-view RRF source-priority order. For each source,
render associated frozen teacher facts once, preserving their source roots. If
the source has no extracted facts, render its original turn. Pack within 3000
cl100k tokens and render chronologically with the original session dates. Raw
ledgers and the original retrieval implementation remain unchanged.

The reader is the same local Qwen3-30B-A3B-Instruct-2507, bfloat16, deterministic
greedy decoding, 192-token cap and original concise-answer instructions. Reuse
the already saved 80 plain outputs. Inputs are prepared without loading QA gold;
scoring is separate. No fact-count, mixture-weight or prompt sweep is planned.
Before scoring, a gain of at least 3 F1 points without truncation is the threshold
for a separate-conversation check; a smaller gain does not justify expansion.

This changes both semantic presentation and packing density. It cannot isolate
normalization from additional source coverage. Provenance-root recall does not
mean that every word of those original turns is visible: extracted facts can
omit details or preserve teacher errors. A source's existence alone does not
establish that the shown paraphrase answers the question.

The 80 answer-free inputs average 2995.9625 context tokens, 77.575 facts, 3.2 raw
fallback turns, and 89.5625 provenance sources. Artifacts:
`runs/phase3/local_canonical_values_dev/`; preparation/scoring:
`scripts/canonical_value_study.py`; representation: `src/canonical_values.py`.
The existing HF generator is used with method `canonical`; CAD and structured
repair are disabled. This is a local reader experiment, separate from the
incomplete SiliconFlow full evaluation.

## Computational sufficiency, rather than information creation

Let X be the immutable conversation, Q the question, W the frozen writer, and
M its question-blind extracted facts. Conditional on W, the Markov chain
Y — (X,Q,W) — M gives the data-processing inequality
`I(Y; M | Q,W) <= I(Y; X | Q,W)`. Extraction cannot create information about the
conversation's target answer that an unrestricted ideal reader could not obtain
from X with the same fixed writer knowledge. This is an explanatory derivation,
not a new theorem or a measured calibration guarantee.

The actual reader has limited compute, context and learned decoding behavior.
A smaller representation can therefore reduce its error even though it loses
information in the Shannon sense. The empirical question is whether the facts
preserve the question-relevant distinctions while removing inference burdens.
The current test checks that end-task utility directly; it does not assume M is
a sufficient statistic. If lost details outweigh easier interpretation, stop
rather than claiming that compression should work by information theory.

## Development result and one selected confirmation

Uniform canonical values complete all 80 questions without truncation:
**58.4042 → 56.9302 F1** (−1.4740 points), 19 wins / 23 losses / 38 ties.
Temporal category F1 improves **55.2388 → 68.1277**, while multi-hop falls
68.1625 → 61.1491 and single-hop 60.4831 → 49.5082. Provenance-root recall rises
92.5855 → 93.8675, illustrating why root coverage alone is not answer utility.
Reject uniform replacement and do not tune packing mixtures.

One post-hoc development selection uses exactly the question-only `TIME_QUERY`
regex already fixed in `src/calendar_memory.py` before this trial. No category
labels, candidate scores or answers enter routing. It selects canonical values
for 27 questions and reuses raw-source outputs on the other 53, producing
**62.5227 F1** (+4.1185), 12 wins / 6 losses / 62 ties. This representation choice
was made after seeing category results; the development score is exploratory.
`reports/local_selective_canonical_exploratory.json` records that status explicitly.

Freeze this one rule and check the existing 160-question, separate-conversation
manifest on conversations 2–9 (seed 20261111). It chooses 47 canonical contexts;
113 candidate outputs reuse the unchanged raw-source branch. Generate the 160
local-model raw baselines once, then the 47 changed inputs. Keep weights, greedy
decoding, answer prompt and 3000-token context budget identical. This is a new
local-model comparison, not a substitution for missing DeepSeek outputs. Those
conversations have been evaluated in earlier research revisions; they are not
an untouched final test. Do not tune the rule after this comparison.

Preparation:
`src/local_cad.py prepare --run local_plain_validation --convs 2,3,4,5,6,7,8,9
--limit 160 --plain-only`, then
`scripts/canonical_value_study.py prepare --run local_selective_canonical_validation
--baseline-run local_plain_validation --temporal-only`.
Generate both runs using the unchanged local HF backend, score the plain baseline
with `src/local_cad.py score --run local_plain_validation`, then score the candidate
with `scripts/canonical_value_study.py score --run local_selective_canonical_validation`.

## Separate-conversation result: stop expansion

All 160 raw baselines and 47 changed candidate outputs completed without
truncation; the other 113 candidate answers reuse their baseline literally.
F1 is **51.5752 → 53.0487 (+1.4735 points)**, 16 wins / 11 losses / 133 ties.
The paired question-bootstrap 95% interval is **[−1.5437, +4.5026] points**.
Temporal category F1 is 53.4861 → 57.8294 (+4.3433), substantially smaller than
the development effect. This does not support the requested larger, reliable
breakthrough. Stop without rule tuning, alternate packing mixtures, or a full
local benchmark run. Report: `reports/local_selective_canonical_validation.json`.

The useful conclusion is narrower: canonical time facts help some questions,
but that effect does not justify replacing original evidence uniformly or
assuming a broad benefit from the selected query rule. The successful DeepSeek
native-thinking comparison remains separate and stronger. No new paid API calls
were made in this study; it does not fill missing DeepSeek full-run outputs.
