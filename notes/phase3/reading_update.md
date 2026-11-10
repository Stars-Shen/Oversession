# Further reading and a local experiment (2026-11-11)

This continues the active research goal after SiliconFlow exhausted its balance.
No paid API queue was restarted. Existing source ledgers and benchmarks remain
unchanged. The following sources were read from their full primary PDFs, with
targeted author-code inspection where noted.

## Temporal memory: useful mechanisms and transfer constraints

**RoMem**, Li et al., EMNLP 2026, pp. 10038–10061:
[paper](https://aclanthology.org/2026.emnlp-main.437/),
[author repository](https://github.com/Tencent/RoMem).
The speed gate predicts relation volatility; target-graph training learns entity,
relation and temporal-spectrum parameters. Its real-valued anisotropic relation
weights break rotational symmetry. Appendix A explains this; simply rotating both
fixed Qwen vectors would leave their dot product unchanged. The advertised zero-shot
transfer concerns the gate, not an absence of target-graph training. Bundled gates
also expect specific text embedding spaces, not any vector of the same dimension.
Reported MultiTQ improvements use a fixed candidate pool, temporal negatives and
trained graph scores; they cannot be attributed to an untrained rotation. Inspected
the author's ChronoR module and requirements. License compatibility was not
established, and no author code was copied or installed.

**CoMMA**, Wang et al., AACL-IJCNLP 2026, pp. 1925–1944:
[paper](https://aclanthology.org/2026.aacl-long.88/).
It extracts entity/slot/value/time/provenance claims, groups candidate conflicts,
and resolves them according to current, historical, provenance or dispute intent.
The controller distinguishes insufficient evidence from incompatible evidence.
Its main 500-item result is on adapted LongMemEval-Oracle, not our cleaned S
retrieval setting: token F1 48.8 versus full-context 44.7, with one main run.
RAG-only retrieves a single memory. Conflict F1 measures explicitly emitted objects;
zero for other systems does not prove they cannot reason about conflict. The paper
acknowledges these limitations. Useful transfer: preserve versions at read time;
do not globally overwrite every differing claim.

**Temporal Semantic Memory**, Su et al., Findings ACL 2026, pp. 29935–29951:
[paper](https://aclanthology.org/2026.findings-acl.1496/).
Actual event time grounds a graph index; monthly semantic clusters yield topic and
persona summaries. Temporal intent filters summaries and promotes supporting raw
turns. Reported GPT-4o-mini LME accuracy is 74.8, using a GPT-4.1-mini judge. The
temporal-component ablation changes overall accuracy by 2.0 points, so the headline
12.2-point baseline gap is not the isolated effect of time handling. The authors
also report that full context wins on LoCoMo with Qwen3-30B: context is only about
16k–26k tokens. Comparisons are not interchangeable with our official F1 and judge.
Fixed monthly aggregation remains a limitation. No code or results were reproduced.

## A concrete front-end limitation

An offline audit of the existing 24-question LongMemEval development memories found
7381 facts and 122 facts explicitly typed as mutually exclusive states. Only two
consolidated tracks contain multiple such state observations, giving two adjacent
pairs. Those inputs cannot support a meaningful learned changepoint likelihood.
Broadening every track into a state would falsely treat events and preferences as
exclusive values. Improve state alignment before testing Bayesian evolution.

Numerical check: a shared unitary rotation leaves a pairwise dot product unchanged
to floating precision. This checks the proposed *naive transfer*, not a purported
error in RoMem, whose trained anisotropic weights avoid that symmetry. Audit
artifact: `reports/temporal_transfer_audit.json`, reproducible with
`scripts/audit_temporal_transfer.py` (both candidate pairs span different sessions).

## Context-aware decoding: a different reading-stage hypothesis

**CAD**, Shi et al., NAACL 2024, pp. 783–791:
[paper](https://aclanthology.org/2024.naacl-short.69/),
[author code](https://github.com/xhan77/context-aware-decoding).
At each output token, use `(1+alpha)*logits(with memory) - alpha*logits(without
memory)`. Normalization gives the same distribution as multiplying the ordinary
conditional probability by a conditional pointwise-mutual-information factor.
The connection to classifier-free guidance allows direct use of the installed
Transformers implementation, with guidance scale `1+alpha`; no author code is
copied. The paper reports gains on summarization and knowledge-conflict datasets,
not conversational memory. Amplifying unreliable context can also amplify error;
prior knowledge remains useful for explicitly requested commonsense inferences.

Local trial: frozen Qwen3-30B-A3B-Instruct-2507, bfloat16, GPUs 1–3, deterministic
greedy decoding, 192-token cap. Compare ordinary decoding with fixed `alpha=0.5`,
without a parameter sweep. Alpha zero algebraically recovers ordinary decoding.
Both modes use the identical 3000-token v2 source-packed context and instructions;
the second branch has the same question/instructions with memory removed. The
generator consumes an answer-free input manifest; official LoCoMo scoring runs
afterward in the separate scoring environment. This is a new local-model result,
never a replacement for missing DeepSeek outputs. Artifacts:
`runs/phase3/local_cad_greedy_dev/`; entry point `src/local_cad.py`.

Setup controls: an initial one-output setup run inherited the model's sampling
defaults through Transformers and was stopped and excluded. The actual experiment
passes an explicit `do_sample=False` override. A tiny-model regression checks this
override and exact recovery of ordinary tokens at alpha zero. The installed MoE
implementation calls all 128 experts even for empty inputs; a process-local
equivalent skips empty experts, preserving the original expert/token order. CPU
and CUDA checks found exactly equal block outputs for decode and prefill shapes.
No installed or shared library file was modified. GPUs are bound by UUID because
CUDA's ordinal order differs from the physical `nvidia-smi` order on this server.

Decision: expand beyond 80 development questions only if CAD produces a useful
answer-metric gain. Stop if it merely suppresses abstentions or damages inference.
Do not present better grounding as established before measuring it.

Reproduce preparation and scoring with `.venv/bin/python src/local_cad.py prepare`
and `.venv/bin/python src/local_cad.py score`. Generation uses the existing torch
environment's Python and `src/local_cad.py generate`, with the physical cards 1–3
bound by UUID in `CUDA_VISIBLE_DEVICES`. The exact UUID binding and generation
overrides are recorded in the run's `runtime.json`. The two local decoding
regressions can be run with that Python using
`-m unittest discover -s tests -p test_local_cad_torch.py`.
