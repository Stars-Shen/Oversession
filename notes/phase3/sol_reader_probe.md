# One stronger reader as a capacity probe

Prospective protocol, 2026-11-11, before generation. Luna medium improves
development F1 only 65.4549 -> 65.9598; stop its effort/budget sweep. Existing
source packing covers most annotated IDs, but neither source-attention learning
nor complete-history exposure gives the desired absolute improvement. Make one
stronger-reader comparison to decide whether this evidence supports materially
better answers. This is not a formal upper bound, a new memory algorithm or an
untouched benchmark. No sibling-model or reasoning-effort search is planned.

Use `gpt-6.1-sol`, medium reasoning, 4096 maximum output tokens, matching Luna
medium's exact 80 question/context/instruction inputs and inference budget.
Only the reader model changes. Reuse the existing Luna low/medium, Qwen plain
and DeepSeek deliberate outputs; do not rebuild memories or regenerate controls.
The strongest existing development point estimate is now **65.9598**, not
65.5748. Require >=3 F1 points above that and <=5% incomplete before a single
separate-conversation check. Do not tune this probe if it fails.

The [official model guide](https://developers.openai.com/api/docs/models)
positions Sol as balancing intelligence and cost, with
[published model rates](https://developers.openai.com/api/docs/models/gpt-6.1-sol).
This is a hypothesis about LoCoMo, not proof of better task accuracy. Sol's
uncached input/output prices are 20 times Luna's; cached-read prices have a
different ratio. It is not claimed cheaper than Luna. Keep Flex (half Standard),
explicitly verify the returned tier, and issue one useful question first.
The complete 80-input public-rate planning estimate, even with all output
limits used, must be <=$3.10 before sending requests. No higher-price Standard
fallback, ambiguous retry, output-limit doubling, GPU allocation or batch upload.
This small capacity probe is separate from the inexpensive default Luna workflow.

Report actual completed-request usage and current-rate cost estimates, including
the pilot once. API/model failures remain visible; do not substitute zero cost
or a partial answer score for a completed experiment. The original shared API,
retriever, data and baseline paths stay unchanged. No local native job resumes.

## Result and subsequent user restriction

The 80-question probe finishes without capping: Sol scores **62.4512**, versus
Luna medium **65.9598**, estimated new cost **$0.332399**. Reject expansion.
Six purposeful cached cases are in `reports/sol_reader_diagnostic.json`; they
are not a prevalence estimate or new evaluation.

The user explicitly requires **gpt-6-luna only, no Sol**. The Flex entry point
rejects other models before request setup. The Sol script is archived in the
ignored run directory and its generation command is retired. No Sol process
or expansion queue remains. Historical prices are accounting records, not
permission to call another model. Subsequent active research uses Luna.
