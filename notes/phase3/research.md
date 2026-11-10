# Phase 3: semantic connections, uncertainty, and evidence dependencies

Research date: 2026-11-11. Active branch: `research/precomem-rgmem-v3`.
The user authorized broad exploration beyond PreCoMem/RGMem, SiliconFlow
`deepseek-ai/DeepSeek-V4.1-Flash`, and local GPUs. No new OpenAI calls.
Stop low-information experiments rather than completing a conventional ablation grid.

## What the strong papers actually do

### PreCoMem

[PreCoMem, arXiv:2610.29467v1](https://arxiv.org/html/2610.29467v1),
2026-10-21; inspected author repository
https://github.com/ZhongWishing/PreCoMem at
`fc2340a23b81b2c4949e80bad227109e4a204bcf` (MIT).
Read retrieval, surprisal signals, lifecycle, correction, agent orchestration,
and signal tests in the existing read-only official checkout.

The reusable mechanism is a belief lifecycle, not merely a retrieval score:
low-confidence hypotheses wait before promotion, compatible observations reinforce
facts, and contradictory observations softly discount old beliefs without deleting
their original evidence. Retrieval mixes semantic, BM25, and topic signals with
belief confidence. The effective surprise combines semantic deviation and maximum
contradiction, gated by normalized score concentration:

`rho = 1 - H(normalize(top-k scores))/log(k)`;
`S = rho * (lambda * semantic_deviation + (1-lambda) * contradiction)`.

Thresholds route maintain/profile/correct. Maintenance approaches confidence one;
correction multiplies by `exp(-beta*S*severity)`. An entropy-regularized proximal
objective explains the exponential update, but this does not establish calibration
of the NLI-derived severity or of the score concentration. Uniform scores can mean
several equally useful witnesses; one sharp score can still be wrong. The paper
itself discusses the proxy and retrieval-budget sensitivity; this is a transfer
limitation, not a claim that the authors missed it.

Reported LoCoMo overall: 73.57 (4o-mini) / 85.06 (4.1); LongMemEval-S:
84.28 / 89.39. These are **paper-reported judge accuracies**, not our F1 or official
LongMemEval judge results. Section 3.1 uses a primary gpt-4.1-mini and two
gemini-2.5-flash judges with different prompts, averaging their binary judgments,
over three seeds. LoCoMo response inputs average 3.3k / 3.1k tokens; LME 4.1k / 3.8k.
Do not rank these numbers directly against our earlier 61.78 F1 or 80.4 official accuracy.

Transfer: separate observation confidence from retrieval-score sharpness. Keep a
waiting room for inferred traits and attribute-local update histories. A change in
one attribute must not invalidate unrelated or historical evidence.

### RGMem

[RGMem, arXiv:2510.16392v1](https://arxiv.org/html/2510.16392v1),
2025-10-18; https://github.com/fenhg297/RGMem at
`637352e7a747d957d3e08201aba06ee4cbda3c4d` (MIT).
Read L1 facade, L2 retriever, and full method/experiments. Existing checkout is
reference only; no services or previous experiments were started.

The paper's consistency/fidelity/complexity Hamiltonian motivates engineering
choices; it explicitly is not a directly optimized computable objective over text.
Three scales incrementally combine relation evidence, node summaries, and ontology
hierarchy. Evidence thresholds gate expensive structural updates. Crucially, the
coarse representation preserves both common structure and residual exceptions or
conflicts. Dirty-node propagation avoids globally rewriting all profiles.

Reported LoCoMo judge accuracy: 78.92 (4o-mini), 86.17 (4.1-mini);
PersonaMem overall 74.01 (4.1-mini). These differ in reader and scoring from ours.
The supplied v1 is treated as an arXiv source; template conference/DOI placeholders
are not evidence of publication. Reproducing its Elasticsearch/Redis/Neo4j service
stack is unnecessary to test the multi-scale mechanism.

Transfer: generate focused cross-session semantic nodes with explicit residuals;
retain immutable low-level facts. Use high-level representations to locate original
witnesses before letting generated abstractions influence an answer.

### HippoRAG 2: concepts should locate context

[Gutiérrez et al., ICML 2025, PMLR 267:21497–21515](https://proceedings.mlr.press/v267/gutierrez25a.html),
arXiv:2502.14802. Full proceedings PDF and author code read;
https://github.com/OSU-NLP-Group/HippoRAG (MIT). Online main was inspected on the
research date; GitHub API rate limiting prevented pinning that snapshot. No code
was copied and no dependency stack installed.

The open graph combines phrase nodes with passage nodes and synonym/context edges.
Query-to-triple matching retains more context than query-entity extraction; an LLM
recognition step filters seed triples. Personalized PageRank retrieves passages,
and the reader receives those original passages rather than generated graph summaries.
The reported average QA F1 is 59.8 versus NV-Embed-v2's 57.0 under a common
Llama-3.3-70B reader across the stated RAG benchmarks (Table 2, arXiv v1).
These are not long-dialog memory benchmark results.

Its own error analysis is useful: in 100 cases with incomplete recall@5, phrase
link failures rise from 7% before to 26% after recognition filtering. This cautions
against treating a selective LLM gate as automatically reliable.

Transfer: explicitly preserve a direct source path; normalize graph propagation so
one broad abstraction cannot award a full vote to each of dozens of children.

### A-Mem and Zep: contextualization and bitemporal structure

[A-Mem, arXiv:2502.12110](https://arxiv.org/html/2502.12110v1): notes contain content,
time, keywords, tags, contextual descriptions, embeddings, and links. New notes can
update contextual attributes of linked notes. This suggests semantic enrichment,
but mutation requires our own immutable provenance guard. Full PDF read, including
method, evaluation, prompts and limitations; code not imported or reproduced.

[Zep, arXiv:2501.13956](https://arxiv.org/html/2501.13956v1): episodes, entities/facts,
and communities form separate graph layers. Event validity and ingestion time are
different axes; contradictions apply to semantically related, overlapping facts.
Hybrid lexical/dense/graph search and reranking construct the final evidence.
Full PDF read. Its LME judge results should not be pooled with our official judge.
This motivated an explicit event/mention-time table experiment below.

## Transfers from other areas

### Database provenance

[Green, Karvounarakis and Tannen, PODS 2007, Provenance Semirings](https://www.cs.ucdavis.edu/~green/papers/pods07.pdf).
Read definitions 3.1/3.2, positive relational algebra, polynomial provenance and
Datalog extension. Label each immutable source with a variable. Alternative
derivations add (`x1+x2`); jointly required premises multiply (`x1*x2`).
Our existing flat list of source IDs cannot express this distinction. A later
dependency-aware memory could keep alternatives and conjunctions separate, so that
one missing witness invalidates a joint derivation but leaves another complete
derivation intact. This is a concrete representation improvement, not a theorem
that provenance will improve QA. Positive algebra alone does not handle unrestricted
negation, open-world absence, numerical aggregation or temporal uncertainty.

### Assumption-based truth maintenance

[de Kleer, 1986, An Assumption-based TMS](https://www.dekleer.org/Publications/An%20Assumption-Based%20TMS.pdf),
Artificial Intelligence 28(2):127–162, DOI 10.1016/0004-3702(86)90080-9.
Read assumptions/justifications, labels, environments, consistency and nogoods.
Labels retain minimal consistent assumption environments under which a conclusion
holds. Missing support and support for a negation are different. This offers a
more precise update mechanism than globally reducing confidence after a conflicting
turn: invalidate only derived profiles whose assumption environments are affected.
Textual entailment and determining a true contradiction remain unsolved front-end
problems; a formal TMS does not solve them by itself. Not implemented yet.

### Bayesian change detection

[Adams and MacKay, 2007, Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742),
Algorithm 1: maintain a posterior over run length. Growth mass is previous mass
times predictive likelihood times `(1-hazard)`; reset mass sums likelihood-weighted
mass times hazard. Normalize and update sufficient statistics.
Transfer hypothesis: infer stable state versus real change within a canonical
attribute, rather than using one global surprise scalar. Independence and likelihood
calibration are major assumptions for repeated paraphrases. Not implemented yet.

### Submodular selection: a tested negative result

[Lin and Bilmes, ACL 2011, P11-1052](https://aclanthology.org/P11-1052/).
Weighted set coverage expresses diminishing returns and avoids paying repeatedly
for already represented evidence. Bundles can require all sources of one fact.
Our actual source/header costs share bytes across overlapping bundles and therefore
are not modular costs: no classic greedy approximation guarantee is claimed.
The tested version lost performance, so no weight/top-k sweep is justified.

### Reasoning-guided retrieval

[Trivedi et al., ACL 2023, IRCoT](https://aclanthology.org/2023.acl-long.557/)
interleaves a derived reasoning sentence and new retrieval because the next search
can depend on previous evidence. Full PDF obtained; targeted method reading only
at this stage, not a reproduced implementation. A promising future version should
form explicit unsatisfied evidence requirements, with bounded search, rather than
run indefinite reflection or trust generated personal facts.

## Controlled experiments and decisions

Frozen input: v2 Luna-written LoCoMo memories and Qwen3 embeddings. New calls:
SiliconFlow DeepSeek-V4.1-Flash only; temperature 0, thinking disabled except for
the explicitly named deliberate control. Official
LoCoMo category-specific F1, categories 1–4, official image captions. Same 80
development questions in conversations 0/1, seed 20261111; 3000 final evidence tokens.
The old v2 completed runs and default OpenAI paths are untouched.

| Hypothesis | F1 | Evidence recall | Decision |
|---|---:|---:|---|
| v2 source-level RRF baseline | 64.1649 | 92.5855 | Frozen control |
| Joint-source weighted coverage/cost | 60.8131 | 88.7393 | Stop; generic cheap coverage misallocates relevance |
| Query-local event/mention-time table + originals | 63.2318 | 92.5855 | Stop; modest temporal/inference gains do not offset single-hop losses |
| Question-blind semantic-node diffusion, original-only answer | 60.7409 | 79.9786 | Stop; abstract connections dilute precise evidence |
| Native DeepSeek thinking, same v2 evidence | 65.5748 | 92.5855 | Modest reader effect; no breakthrough claim |
| Local query/window cross-encoder | 65.6405 | 92.6923 | Development-only positive, failed separate validation; stop |

The table compiler consumes the initial 3000-token retrieved context and adds a
separate call. It then retains at most 800 derived tokens plus re-packed originals
within 3000 tokens for the final reader. Thus final context budget is matched;
**total LLM processing cost is larger**. Its recorded source recall includes cited
table witnesses; retained original IDs are recorded separately. Citation membership
is checked mechanically but this is not a semantic entailment certificate.

Graph trial: summaries are built from facts only, without questions, answers or
evidence labels; pattern nodes need support in two distinct sessions. Exceptions
remain linked. Query-node matching and source-level RRF seed a bipartite random
walk with restart; both edge directions conserve mass. Only original turns enter
the answer call. This adapts mechanisms, and is not an official RGMem/HippoRAG
reproduction. Cache and result artifacts live only under `runs/phase3/`.

No result from these small development experiments establishes a breakthrough or
superiority to the cited papers. Any selected method must be tested on separate
conversations; end-to-end SiliconFlow writing is a distinct subsequent question.

The native thinking control uses the exact same answer instructions and original
evidence as v2. Mean recorded reasoning usage is 283.525 tokens/question. This
isolates extra model computation rather than claiming a memory improvement.
Its maximum completion budget is 4096 versus 768 for nonthinking (one doubling
retry on truncation). Total inference compute is explicitly not matched.

The local reranker is [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
(Apache-2.0 weights), downloaded from the official Qwen organization. It scores
the documented yes/no likelihood with a conversation-specific instruction. Input
windows contain the candidate turn and same-session immediate neighbors; original
windows are selected atomically and deduplicated under the final budget. This
tests semantic answer support and dialogue completeness, unlike the failed
coverage objective's gain-per-token. No fine-tuning or model parameter sweep.
Model weights are pinned by downloaded Hugging Face metadata to commit
`e61197ed45024b0ed8a2d74b80b4d909f1255473`, SHA-256
`27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b`.

Separate validation on 160 questions in conversations 2–9 rejected the reranker:
59.6828 to 56.1834 F1. Native thinking, which had been independently specified
before those validation results, improved the same cohort from 59.6828 to 65.4000
(+5.7172 points), with gains in all four categories. The native reader was frozen
for the full 1540-question run. The full report also excludes all development
conversations and those 160 validation questions for a stricter confirmation cohort.

## Interrupted full run: evidence available, no full-benchmark claim

SiliconFlow returned HTTP 402 (insufficient account balance). The run was stopped
with 1855/3080 outputs saved: 931 nonthinking and 924 thinking, forming 924 complete
pairs. A subsequent balance guard stops new HTTP calls across both reader instances
and cancels queued evaluations on the first 402; a regression test covers that rule.
Before that guard existed, the submitted queue produced 1017 rejected 402 calls.
All successful API outputs remain cached; no further model jobs are running.

The completed paired portion improves from 60.5472 to 64.8421 F1. It is **not** a
1540-question score. Removing both development conversations and the 160 validation
questions leaves 603 complete confirmation pairs: **60.0379 to 63.9385 (+3.9006
points)**, paired bootstrap 95% interval **[+1.5912, +6.3301]**, 141 wins / 110
losses / 352 ties. This cohort is the available portion of a fixed evaluation
interrupted by balance exhaustion, not a separately completed full benchmark.

On these 603 pairs, temporal F1 improves 58.1467 to 70.6831 and inference 20.7034
to 29.5039; single-hop declines 75.3477 to 71.3285. This supports a reading/use-stage
bottleneck, particularly for time reasoning, but does not establish a new memory
algorithm or uniform improvement. No remaining questions were filled with another
model. Full 1540-question evaluation still requires the same provider to be funded.

Eight local tests pass, including provenance filtering, atomic evidence-window
packing, balance failure handling, and the unchanged baseline regressions.
