# One fixed same-size native reader replacement

Prospective protocol, 2026-11-11, before generation or scoring. The complete-history
treatment fails; the current source adapter trails the strongest reader on the
same 160 inputs in every category. Another context-order/length grid or 1147
more adapted answers has low decision value. Cancel the queued adapted expansion,
preserve all existing caches, and let the original control finish as a useful
common baseline. It is not a completed paired full evaluation.

The next hypothesis is that a 30B native reasoning reader can use the existing
3K evidence more effectively. The earlier 4B native result beat its own weaker
control but not the 30B Instruct reader; it does not settle this same-size test.
Use official [Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507),
Apache-2.0, pinned revision `144afc2f379b542fdd4e85a1fcd5e1f79112d95d`.
Its architecture is supported by the existing Transformers 4.51.3 environment.
Download 61,064,245,248 weight bytes to the task-local ignored model directory;
streaming SHA256 checks use pinned Hugging Face LFS metadata. No new framework,
vLLM repair, training, paid calls or shared-weight edits.

Copy the exact 80-question original 3K answer-free manifest. Keep the system
instructions, question text, context, source IDs, writer and retrieval unchanged.
This checkpoint supports only native thinking. The model weights and inference
policy change together, so this is a reader-replacement experiment, not causal
attribution to thinking alone or a new memory algorithm.

Use the card's temperature .6, top-p .95, top-k 20, min-p 0, with one fixed seed
per batch derived from its question IDs and seed 20261111. No repeated seeds or
sampling grid. The hard completion budget is 4096 tokens, the same practical
budget as the earlier local native study, not the much larger model-card math
budget. Its adequacy is an empirical limitation: unfinished thoughts become
abstentions and remain in scoring; no automatic budget increases. Parse only
the text after the last native closing-thought token using the already tested
parser. Keep reasoning separately; it never becomes an answer for scoring.

Twenty fixed batches of four questions alternate between two GPU replicas.
Per-batch seeds and membership are fixed before generation. This is one seed
per batch, not independent per-question sampling or a multi-seed result.
Physical GPUs 1/2/3 use explicit 16/16/16 layers after the old control exits;
4/5/7 use the successful 16/23/9 placement. This avoids per-token scheduling
overhead without changing the attention computation or installing new kernels.
No memory-profile search. Both processes write separate files.

Score only after all 80 exact question IDs exist. Compare with existing original
Qwen 3K (58.4042) and strongest DeepSeek deliberate (65.5748), retaining all
incomplete outputs. Expansion requires >=3 F1 points over the latter and <=5%
incomplete cases. If not, stop this budgeted reader policy without trying a
temperature, prompt, length or reasoning-budget grid. Reused development is not
an untouched benchmark. A useful result still needs separate-conversation and
full-evaluation evidence before a larger final breakthrough is claimed.

Artifacts: `src/native_30b_reader.py`, `runs/phase3/native_30b_dev/` and eventually
`reports/native_30b_dev.json`. Initial standard Hub download failed on proxy HEAD
metadata, before any model execution. `scripts/fetch_qwen30b_thinking.py` uses
verified streaming GET instead; this is infrastructure recovery, not an F1 retry.

A finite coordinator is running at `runs/phase3/native_30b_dev/pipeline.py`.
It waits on the confirmed-live pinned download, checks the current free memory
of the six specified GPU UUIDs, launches exactly the two 40-question generation
processes, and calls the existing CPU scorer once both succeed. Its current
phase, coordinator PID and generation PIDs are in `pipeline_status.json`;
logs are `pipeline.log` and `generation.0.log` / `generation.1.log`. It queues
no separate-conversation or full-evaluation expansion. SIGTERM to that owned
coordinator stops its owned generation children while preserving saved batches.
No unrelated GPU job is stopped. A download/generation failure is recorded as
infrastructure failure, with no automatic model restart or F1-policy change.

This note documents the active authorized task, not an instruction for another
session to resume it automatically.
