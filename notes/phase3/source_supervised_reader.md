# Source-supervised reading: fixed hypothesis and protocol

This experiment asks whether training a competitive reader to use complete
supporting context can improve answers without adding inference calls or an
evidence-output schema. It follows the failed 4B answer-only adapter; it is not
another parameter setting of that candidate. Frozen 30B Qwen3-A3B weights,
memory, RRF retrieval, 3000-token context, prompt and greedy 192-token answer
decoding remain identical between control and treatment.

## Evidence behind the mechanism

Targeted methods/results reading of [QEMem, Li et al., 2610.22668v1](https://arxiv.org/html/2610.22668v1)
finds query planning, multi-granularity retrieval, source-linked ViewCards and
incremental gap feedback. It reports better complete annotated-source coverage
and Claude-judged answering, with extra calls. A quote substring check validates
text presence, not entailment. Its judge scores are not our official F1. The
useful transfer here is joint evidence sufficiency; static cards alone already
failed in our experiments. No author implementation is imported.

[EviCover, Roshchupkin and Khodorchenko, 2611.10170v1](https://arxiv.org/html/2611.10170v1)
is a useful negative diagnostic. On REALTALK, its final ten-turn evidence recall
rises .349 to .379 and complete sets 36 to 38, at 17.47 additional model calls
and 24.59K additional tokens per question. Reader prompt changes can improve
grounding while F1 falls .183 to .147. That does not justify an expensive
setwise-selector sweep here. Coverage is a proxy; answering is the decision.

[Ross et al., Right for the Right Reasons, IJCAI 2017](https://www.ijcai.org/Proceedings/2017/371)
regularizes input-gradient explanations using annotations. Our transfer uses
cheap first-order attention coverage instead, not their literal objective or
reproduction. [Jain and Wallace, Attention is not Explanation, NAACL 2019](https://aclanthology.org/N19-1357/)
provides the relevant limit: attention weights need not faithfully explain
predictions. A falling auxiliary loss is not a faithfulness result.

## Fixed training and decision

Only conversation 0 supplies supervision. Of its 152 questions, 131 have all
annotated sources visible in the original packed context; 25 require multiple
sources. Train on those 131, with final-answer targets and their source IDs.
Conversation 1's 81 generation inputs contain neither targets nor gold sources.
No question from conversations 2–9 contributes supervision.

Use rank-8, alpha-16 query/value LoRA, three epochs, learning rate 1e-4,
accumulation 8, clip 1, seed 20261111. Answer CE is normalized by total target
tokens in each accumulated batch, including EOS. Add weight .1 times the mean
positive-source coverage loss. This is a joint learned-reader intervention,
not an isolated causal estimate of the auxiliary term or normalization.

Coverage uses the actual final layer's query at the last prompt token, before
any gold answer. Reconstruct its normalized Q/K, rotary positions, GQA and
softmax over the causal prefix. For every positive source, sum attention over
its text tokens and take the maximum across heads; minimize mean negative log
mass. Different required sources may use different heads. Exclude source IDs,
speaker names and session headers from each source's content span. Other heads
remain free; unannotated sources receive no explicit negative labels, although
softmax competition still implies gradients. This is not an entailment check.

Tiny-model tests verify actual eager attention equality, no future-target
visibility, gradient flow and unchanged input gradients from skipping empty
experts. Perform one maximum-length real 30B backward smoke, without an optimizer
step. Save only task-local adapter tensors; original model files remain intact.

One final adapter, one 81-question validation. Reuse 20 exact existing control
answers and generate 61. Expand to the existing 160-question, eight-conversation
cohort only for at least +3 paired F1 points with at most 5% capped answers in
each condition. A meaningful separate-conversation gain can justify one full
1307-question evaluation. No rank, epoch, auxiliary-weight or prompt grid.
These cohorts have been reused in research; do not call them untouched tests.

Before any validation result is available, the later separate-cohort report also
records a target-category-mixture sensitivity. It weights each observed category
effect by that category's share among the full 1307 eligible questions, following
the earlier [native-reasoning mixture audit](category_mix.md). This adds no
generations or extra cohort. It is not an observed full score: within-category
effects may also vary across questions/users. Keep the declared triage gate
unchanged and use this information when deciding whether a full evaluation has
enough decision value; do not promote a balanced-cohort gain to a full result.

## An adjacent alternative, read but not launched

Targeted method, tables, limitations and prompts of
[Wang et al., E-mem, 2601.21714v1](https://arxiv.org/html/2601.21714v1)
show a complementary mechanism: overlapping intact episodes, summary/raw/sparse
routing, small-model local evidence and central synthesis. Local output includes
verbatim text plus a preliminary answer. Iterative querying and persistent KV
states are extensions; their presence in the description does not establish
their contribution to reported gains. LoCoMo F1 is 54.17 with GPT-4o-mini and
57.04 with Qwen2.5-14B; those reader/budget settings differ from ours. The 8K
chunk study uses about 32K total context, not our 3K retrieval budget. Its 70%
saving against GAM concerns a 1:10 weighted small/large-token cost proxy; total
tokens are 2406 versus 1254. Reasoning latency can increase. Preserving original
episodes is a useful transfer, but copying XML or naming more agents does not
establish causal integrity. No E-mem implementation or multi-agent job is run.

Entry points: `src/source_supervised_reader.py` prepare, train,
prepare-evaluation, score; generation uses `src/local_cad.py` on
`local_source_control_validation` / `local_source_validation`. Separate
preparation/scoring is available only for a candidate passing that decision.
Training artifacts live in `runs/phase3/source_reader/`.

## Completed training and first validation

The fixed training completed all 51 steps / three epochs in 4121.0 seconds,
with 3,342,336 trainable parameters. Peak allocated CUDA memory was
20.10/20.68/19.04 GiB. Final adapter SHA256:
`5d44fe88809408e7f4e1722519a25ef133cef267135e0763f6b9a43dcf2059d1`.
The earlier maximum 3465-token backward smoke passed without an optimizer
update. Original model weights remain intact.

All 81 matched validation pairs complete: **58.6216 -> 61.1989 F1 (+2.5772
points)**, 21 wins / 19 losses / 41 ties, question-bootstrap interval
[-2.0289, +7.2408] points. Neither condition has capped answers. Category effects
are multi-hop +1.7103 (11 questions), temporal +2.7040 (26), and single-hop
+2.7191 (44); there are no inference-category questions. This remains a
one-dialogue result. The declared +3-point triage gate is false in
`reports/source_reader_validation.json`.

An attempted concurrent control on heterogeneous remaining GPU memory triggered
the no-CPU-offload guard before any new answer. That allocation path was stopped
and its extra configuration removed. The original twenty controls remain intact.
The task-local evaluation queue completed both sets of answers on the original
three GPUs. Its final scoring command failed because the Torch environment
lacks NLTK. Scoring then completed once with the existing `.venv` CPU environment;
no answer generation was repeated and no dependency was installed. The recovery
and original error are recorded in `pipeline_status.json`.

## Explicit one-time triage exception, before separate-cohort results

The original gate remains false. The user accepts rough intermediate steps and
prioritizes final-metric decisions. Because the single validation dialogue shows
positive effects in every observed category but provides no inference coverage,
make one explicit exception: check the unchanged final adapter on the existing
160-question, eight-conversation manifest, including all four categories. This
changes neither weights nor prompt, retrieval, decoding, candidate selection,
or training schedule. It is not an epoch/rank/weight search, nor a claim that
the declared gate passed.

For a full 1307-question evaluation, require at least +3 paired F1 points on
that cohort, at least +3 under the target-mixture sensitivity, and at most 5%
capped answers. There is no further triage exception for this candidate. The
160 questions have been reused in research; do not call them untouched tests.
No full run or paid call is launched by this decision.

This note records the active authorized experiment. It does not instruct another
session to resume it automatically.
