# Task risk, candidate decisions, and the metric that is actually measured

## Translation: two distinct transferable mechanisms

Shen, Cheng, He, He, Wu, Sun and Liu, [Minimum Risk Training for Neural Machine
Translation](https://aclanthology.org/P16-1159/), ACL 2016, pp.1683–1692,
DOI 10.18653/v1/P16-1159. Targeted methods, results and failure analysis reading:
minimize expected sequence loss on sampled candidates, with a normalized
`P(candidate)^alpha` distribution; their sampling set includes the training
reference. The centered loss term yields a score-function gradient without
differentiating BLEU. Chinese/English, English/French and English/German results
improve over their likelihood-trained NMT controls. It uses 100 samples and
substantial training compute; the smoothing parameter matters. Long outputs
remain limited by the training length cap. Transfer: optimize final answer
quality using training-only references, rather than assume token CE directly
optimizes complete lists. This is a possible subsequent hypothesis, not a
second training job or a reproduction of their NMT system. No paper code is used.

Freitag, Grangier, Tan and Liang, [High Quality Rather than High Model Probability:
Minimum Bayes Risk Decoding with Neural Metrics](https://aclanthology.org/2022.tacl-1.47/),
TACL 10, 2022, pp.811–825, DOI 10.1162/tacl_a_00491. Targeted sections 3, 5–7:
select the candidate with highest mean utility against model-generated
pseudo-references. The main translation experiments use 1000 unbiased samples
and BLEURT; smaller-pool studies also show gains. Repeated samples retain their
multiplicity. Model distribution error and utility validity remain separate
failure modes. Neural-metric human-quality gains do not establish that lexical
consensus improves memory QA. A heterogeneous cached committee can test a
cheap decision rule, but is not iid sampling or a calibrated answer posterior.

## Multi-label decision theory: useful derivation, mismatched direct deployment

Dembczynski, Jachnik, Kotlowski, Waegeman and Huellermeier,
[Optimizing the F-Measure in Multi-Label Classification: Plug-in Rule Approach
versus Structured Loss Minimization](https://proceedings.mlr.press/v28/dembczynski13.html),
ICML 2013, PMLR 28(3), pp.1130–1138. Targeted sections 3–8 derive optimal binary
label-set predictions from joint label/cardinality probabilities. For a fixed
predicted cardinality k, select k labels with largest
`Delta(i,k) = sum_s 2 P(Y_i=1, |Y|=s)/(s+k)`; then compare cardinalities,
including the empty set. No label-independence assumption is required. Their
plug-in consistency statement concerns suitable distribution estimation and
asymptotic optimization, not an arbitrary language model's confidence. Experiments
use six finite multi-label datasets, not QA. Direct deployment here lacks a
fixed semantic-label universe, calibrated joint probabilities and a matching
set utility. Do not call a 0.5 cutoff, threshold grid, or sampled-answer union
an exact consequence of this result.

## Check the target before optimizing its surrogate

The unchanged official `third_party/locomo_evaluation.py:f1` averages, over gold
comma-separated items, the best Porter token F1 among predicted items. This is
directional and is not ordinary symmetric set F1. Non-multi-hop uses multiset
token F1. A synthetic check against reference `guitar, painting` gives:

| Prediction | Multi-hop score | Ordinary token score |
|---|---:|---:|
| guitar | .5 | .6667 |
| guitar, painting | 1 | 1 |
| guitar, painting, unsupported extra | 1 | .6667 |

Those are synthetic metric semantics, not benchmark gains. The multi-hop metric
does not penalize unrelated extra comma items in this example. The official
scorer and baseline remain unchanged. A useful improvement must still preserve
supported answers; exploiting this property would not establish better memory.
Consequently, the binary-set Bayes guarantee cannot be copied to our benchmark,
and unrestricted list union is not selected as a solution.

## One cache-only decision test, fixed before scoring

Use the same 80 development questions and identical frozen source context from
three cached readers: DeepSeek native thinking, DeepSeek without thinking, and
plain Qwen3-30B. For each existing answer, average ordinary symmetric Porter
token F1 against the other two answers; select the highest, breaking ties toward
the strongest existing native-thinking reader. Do not use gold categories,
answers or scores in selection. Duplicate answers count as separate committee
votes. This is a finite-committee medoid, not a posterior-Bayes claim.

Only after saving selected generation fields load true references and evaluate
with the unchanged category-specific official scorer. Compare to the strongest
native-thinking answer, not merely the weaker plain readers. Require at least
+3 paired F1 points to expand once to the existing 160-question cohort. No
weight, utility, model-pool or tie-break search. An oracle pool maximum is only
a diagnostic bound and cannot be reported as achieved F1. Entry point:
`scripts/committee_consensus.py`; no new API/GPU calls. The ongoing 30B source
supervision experiment is unchanged.

## Completed cached result: modest development gain, stop the selector search

All 80 matched cases complete: strongest DeepSeek deliberate **65.5748 →
67.0713 F1 (+1.4965 points)**. There are 12 wins, 9 losses and 59 ties; the
question-bootstrap interval is [−2.7900, +5.6374] points. The selector chooses
the deliberate reader 51 times, nonthinking DeepSeek 19, and local 30B 10.
The declared +3-point expansion gate is false. Do not expand this candidate,
search weights/metrics/tie-breaks, or label the development gain a breakthrough.
`reports/committee_consensus_dev.json` records the result. Selection is saved
before gold scoring in `runs/phase3/committee_consensus_dev/selected_answer_only.json`.

The gold-only pool ceiling is 73.8905, suggesting the pool sometimes contains
better answers than the strongest reader, while this fixed consensus rule
captures little of that potential. That ceiling is not an achieved result or
an inference-time decision rule. Cached analysis makes no new calls; deployment
would require both DeepSeek modes and a local reader, so it is not a free
accuracy gain or a demonstrated cost reduction. This outcome does not establish
that properly sampled neural-utility MBR or training-time MRT would fail.
