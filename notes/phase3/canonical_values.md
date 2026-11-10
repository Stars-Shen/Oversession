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
