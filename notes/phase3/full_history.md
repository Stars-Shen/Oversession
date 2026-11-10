# Full raw history with causal prefix reuse

Prospective protocol, 2026-11-11, before generating or scoring this candidate.
The active goal is a larger absolute metric improvement, not merely a gain
against a weaker reader. On the existing same-context 160 questions, the source
adapter reaches 55.6961 F1 versus DeepSeek deliberate's 65.4000: **−9.7040
points**. Source IDs and context token counts match. Its +4.12 improvement
against its own local control is insufficient to establish that goal.

Hypothesis: the 3000-token retrieval budget is an information bottleneck;
original conversations may be short enough to read directly with the existing
unadapted Qwen3-30B-A3B-Instruct-2507. This changes information exposure, not
the trained reader, answer instructions, greedy decoding or 192-token limit.
Use all frozen raw turns, official image captions included as before, preserving
the existing chronological session-header serializer. No learned adapter, gold
source filtering, synthetic summary or new API calls.

The adjacent primary literature is [TSM](https://aclanthology.org/2026.findings-acl.1496/)
and [E-mem](https://arxiv.org/html/2601.21714v1), read in the existing review.
TSM's full-context Qwen result motivates testing the local information ceiling;
its reported judge metric is not our official F1. E-mem motivates intact
episodic information and mentions latent KV reuse; neither paper proves the
result of this proposed experiment. This is a direct local test, not a claimed
implementation of either paper.

Reuse the fixed 80 development questions from conversations 0/1. Cached controls
are original Qwen with 3K retrieval (58.4042 F1) and the strongest existing
DeepSeek deliberate reader (65.5748). Expand once only if the candidate exceeds
the latter by at least 3 points. A weaker-control gain alone cannot pass.
Do not sweep prompts, lengths, layer profiles, or adapters. All capped/empty
outputs remain in the score. Report point estimates and paired uncertainty;
development has been reused and is not an untouched benchmark.

For each conversation, prefill a prefix computed only from system instructions
and memory. Check that every complete question prompt begins with those token
IDs. After each answer crop the mutable KV cache to that exact length, discarding
all question/answer states. A tiny original Qwen MoE CPU test confirms that two
successive answers equal independent full-context generation and do not change
the retained prefix states. This tests cache semantics, not F1. Numerical
rounding on other hardware need not be bitwise identical.

Prefix reuse exploits causal invariance; it does not create new information or
itself explain an accuracy gain. Count one conversation prefill plus question
suffix work separately from conceptual full input tokens. These are local GPU
work counts, not billable API savings. The full model uses explicit 16/23/9
layer placement on currently available physical GPUs 4/5/7; no CPU offload or
automatic memory-profile grid. Leave the existing paired 3K study on GPUs 1–3
running. An infrastructure failure is not a negative hypothesis result.

Generation inputs contain no gold answers. The CPU scorer loads references
only after all 80 outputs exist. Original shared retrieval/scoring code and
model weights stay unchanged. This protocol records the currently authorized
task and does not instruct a later session to resume any unfinished run.

## Systems and long-context transfer check

Targeted primary sections of [CacheBlend, Yao et al., arXiv:2405.16444v2](https://arxiv.org/html/2405.16444v2)
(methods, evaluation, limitations) distinguish exact prefix reuse from merging
independently prefetched non-prefix chunks. The latter loses cross-chunk
dependencies; CacheBlend selectively recomputes states to approximate full
prefill. Its v2 evaluates Mistral-7B, Yi-34B and Llama-70B on QA/summarization,
with typical quality loss rather than an exact-equality theorem. We do not
copy its code or install its vLLM modifications. Our complete chronological
conversation is one unchanged prefix, so exact causal reuse is simpler and
preserves all within-history attention. This systems idea makes the information
test practical; it is not an independently new accuracy method.

[Lost in the Middle, Liu et al., arXiv:2307.03172v3](https://arxiv.org/html/2307.03172v3)
(Sections 2–5) supplies the competing hypothesis: readers may fail to use
additional relevant context. The controlled NaturalQuestions experiments show
positional sensitivity and recall improving beyond reader performance.
Duplicating the query before/after context helps synthetic key retrieval much
more than natural QA. Thus neither more tokens nor a question-placement trick
is guaranteed to improve this different 2025 Qwen backbone on LoCoMo. Do not
expand a placement/order grid automatically if this fixed test fails.
