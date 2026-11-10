# SiliconFlow research phase

The v2 OpenAI results remain historical artifacts. New phase-3 model calls go only
to `https://api.siliconflow.cn/v1/chat/completions`, using the exact model
`deepseek-ai/DeepSeek-V4.1-Flash`. The model name was verified against the provider's
model listing and a successful completion. [Provider API documentation](https://docs.siliconflow.cn/docs/api/chat-completions-post).

Supply `SILICONFLOW_API_KEY` through the environment or the private ignored file
`.private/siliconflow_key`. Do not put credentials in run arguments or tracked files.

```bash
# Same fixed memory, prompt, questions and final evidence budget; reader mode differs.
.venv/bin/python src/phase3_experiment.py \
  --convs 0,1,2,3,4,5,6,7,8,9 --limit 0 \
  --methods v2,deliberate --budget 3000 --workers 12 \
  --run sf_deliberate_full

# Reconstruct reports locally from existing outputs, with no generation or judge calls.
.venv/bin/python scripts/report_phase3.py
```

`v2` is source-level four-view reciprocal rank fusion with native thinking disabled;
`deliberate` uses exactly the same evidence and instructions with thinking enabled.
Completion limits are 768 and 4096 respectively, with one doubling retry when
truncated. Their actual reasoning and completion usage are reported. This is a
reader-compute comparison, not a matched-total-compute comparison.

Both variants retain the already-built Luna memory and Qwen3 embeddings. No old
extraction/consolidation scripts need to run. Do not describe this as an end-to-end
DeepSeek memory writer benchmark. The original full v2 report is still
`reports/final_v2.json`; the new report is `reports/phase3.json`.

The scorer executes the pinned official LoCoMo functions locally. Categories 1–4
are included, category 5 is excluded. Image captions follow the unchanged v2
protocol. No LLM judge or gold reference enters retrieval or generation.

Exploratory methods `coverage`, `table`, `graph`, and `cross_rank` are preserved
to document the tested hypotheses. They were rejected in development or separate
validation. They are not selected production replacements. The graph was generated
question-blind; local Qwen3 reranking used GPUs 1–3. Their cached evidence and logs
are under `runs/phase3/`, separate from all earlier benchmark outputs.

Development: 80 questions in conversations 0–1. Separate validation: 160 questions
in conversations 2–9. Full reporting separates the 1307 questions outside the
development conversations and the 1147 also excluded from validation. These are
unseen to phase-3 selection, not untouched by the entire earlier project.

Research interpretation and primary references: [research.md](research.md).
One unimplemented follow-up hypothesis: [next_hypothesis.md](next_hypothesis.md).
