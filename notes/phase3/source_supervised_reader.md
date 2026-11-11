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

## Separate-conversation result and full evaluation

All 160 paired questions across eight conversations complete: **51.5752 ->
55.6961 F1 (+4.1209 points)**, 36 wins / 38 losses / 86 ties. The question-bootstrap
interval is [-0.8788, +9.2284] points; both readers have zero capped answers.
The prespecified target-mixture sensitivity is **+4.2851 points**, not an
observed full score. Both expansion point criteria pass, so evaluate the same
final adapter on all 1307 eligible questions. There is no retraining or routing
change. `reports/source_reader_separate.json` records this decision's evidence.

Effects are heterogeneous: multi-hop **-3.9657**, temporal **+3.5604**, inference
**+10.3233**, single-hop **+6.5654** points, with 40 questions per category.
The result does not support a claim of improved multi-source composition.
It supports an overall gain for this joint reader update on this cohort;
the auxiliary loss, answer supervision and normalization are not isolated.

The full comparison excludes both training/validation dialogues. Its
`local_source_control_full` and `local_source_full` manifests use identical
original prompts, 3000-token evidence and greedy 192-token answer decoding.
Each reuses its 160 existing answers after exact input equality checks; only
1147 additional answers per condition are generated. Report both all 1307
pairs and the remaining 1147 after excluding the expansion cohort. Those
remaining questions were not used to choose this adapter, although this dataset
was used in earlier project research. Do not call them globally untouched.

`src/source_supervised_reader.py prepare-full|run-full|score-full` implements
the finite comparison. Run the orchestration with the existing CPU `.venv`;
it uses the existing Torch interpreter only for generation. Physical GPUs 1–3
are reused sequentially, with no new paid calls. Progress is stored in
`runs/phase3/source_reader/full_pipeline.log` and the two full-run generation
logs. The full result is pending; this is not yet a full-benchmark breakthrough.

## Cache-only interpretation and comparison anchor

`reports/source_reader_diagnostic.json` uses only existing outputs and packed
inputs, with zero new model calls. Multi-hop median completion length falls
10.5 -> 5.5 tokens; single-hop falls 8 -> 5. Inference-category occurrences of
the literal standard abstention marker fall 8 -> 0. These observations do not
isolate why training changes answers or establish a causal effect of length.

Five purposeful high-gain cases show a mixture of changes. The packed input
contains both Sam's earlier hiking habit and a December hiking trip when the
answer changes from running to hiking (`conv-49:72`). Other changes replace
abstention with a reference-matching location, with Italy explicit in a dated
source (`conv-47:82`) or geographical clues in the source (`conv-48:28`,
`conv-48:48`). A relative-year answer changes to its clock-resolved absolute
year (`conv-41:53`), which is a normalization gain without independently new
factual content. These are examples, not a prevalence estimate or proof that
the model's attention faithfully explains its decisions. Full-evaluation policy
and weights remain unchanged.

The completed v2 Luna reference on the **same 1307 questions** is **61.4174 F1**,
not its 1540-question aggregate. A one-time alignment check confirms identical
question-ID sets, dataset/scorer hashes, the eight memory-ledger hashes, and
the 3000-token evidence budget. Shared `experiment.py`, `memory.py` and
`longmem.py` are unchanged since the pre-phase3 commit. Comparing with Luna
assesses the complete reader solution; attribution of the training update still
requires the full paired Qwen control. No old API answers are regenerated.

This note records the active authorized experiment. It does not instruct another
session to resume it automatically.

The stronger existing same-context comparison must also be retained. On the
same 160 question IDs, source IDs and context token counts, DeepSeek without
thinking scores 59.6828 and deliberate scores **65.4000**, versus **55.6961**
for this local adapter. The latter is **9.7040 points below** the strongest
reference, which leads in all four categories. The local paired gain does not
constitute the desired larger absolute breakthrough. This is recorded in
`reports/source_reader_diagnostic.json` without new model calls.

## Expansion cancelled after strongest-reference comparison

Cancel the queued remaining 1147 adapted generations: the same-context candidate
trails DeepSeek deliberate in every category, so extending that comparison has
low decision value for the larger-breakthrough goal. The original full control
is allowed to finish as a reusable baseline. Its orchestrator is paused solely
to prevent launching the queued adapted child; terminate that parent after the
control finishes. Preserve the 160 adapted outputs, checkpoint and all baseline
outputs. `full_pipeline_status.json` explicitly records intentional cancellation,
not successful full pairing. No completed full adapter result is claimed.

The original control subsequently completes all **1307** questions at **57.6954
F1**, with one capped output retained. The same-cohort old Luna reference is
61.4174. `reports/local_original30b_full.json` records this completed original
reader result and the separate cancelled-adapter status. There is no observation
of the adapted reader's full score and no extrapolation is substituted for it.
