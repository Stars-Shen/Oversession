# Versioned Evidence Memory (research workspace)

Goal: improve extraction, representation, evolution, retrieval and use of long-dialog memory on LoCoMo and LongMemEval. Results are experimental; this is not yet a reproduced SOTA system.

The memory writer receives only dated conversation turns. It produces self-contained atomic facts with source IDs. Raw turns stay immutable. Mutually exclusive current-state observations can form chronological version chains; historical events and multivalued preferences stay additive. Retrieval combines raw evidence and fact indexes, optionally local Qwen3 embeddings, within a fixed token budget. Answers and scoring are separate from the writer.

## Reproduce current experiments

Use `.venv/bin/python` for scripts except embedding. Python 3.11 packages are pinned in `requirements.lock.txt`. API calls use exactly `gpt-6-luna`, Responses API, low reasoning. Set `OPENAI_API_KEY` or use the private local credential file; it is excluded from version control. `MEMORY_PROXY` defaults to `http://127.0.0.1:41227`.

```bash
.venv/bin/python src/experiment.py build --convs 0,1 --workers 8
.venv/bin/python src/experiment.py evaluate --convs 0,1 --limit 80 --methods raw,facts,evolved --run pilot_v1
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 /home/wqshen/anaconda3-new/envs/user-memory-study/bin/python src/embed.py --convs 0,1
.venv/bin/python src/experiment.py evaluate --convs 0,1 --limit 80 --methods raw,hybrid,hybrid_facts --run hybrid_dev_v1
.venv/bin/python src/longmem.py build --per-type 4
.venv/bin/python src/longmem.py evaluate --per-type 4 --methods raw,facts --run longmem_dev_v1
```

LoCoMo conversations 0 and 1 are development; 2-9 are held out. Non-adversarial categories 1-4 use unchanged official stemmed F1 functions loaded from the upstream scorer, including comma-split multi-hop answers and category-3 first reference. A fixed seed selects the development questions. Gold evidence is used only to calculate recall after retrieval. F1 scores across different papers' reader models and protocols are not directly comparable.

LongMemEval uses cleaned S histories, not oracle histories. All `has_answer` fields and original session IDs are excluded from writer inputs; source IDs are local positions because upstream IDs can contain answer/noans prefixes. The first four questions per category sorted by SHA256 are development. The current development judge uses unchanged official prompts with Luna, so its score is **not** the official GPT-4o judge score.

Run configs include dataset, prompt, scorer and memory hashes. Predictions, API outputs, usage and summaries are under `runs/`. Cached calls allow a stopped experiment to resume. A config mismatch requires a new run name. Source notes, provenance and hypotheses are in `notes/research.json`.

## Initial evidence

On 80 development questions at 3,000 context tokens: raw BM25 F1 41.80; atomic facts 48.44; facts plus naive adjacency/state closure 46.51. State chains are absent in these two development conversations, so this experiment does not validate knowledge-update evolution. The naive closure is not selected for expansion. Stronger hybrid retrieval and a reader using shorter answer phrases are being measured.
