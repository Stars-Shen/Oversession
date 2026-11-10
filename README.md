# Versioned Evidence Memory

Research workspace for a five-stage long-dialog memory system. Full benchmark runs are in progress; current results do not establish a SOTA claim.

| Stage | Implementation |
|---|---|
| Extraction | Question-blind atomic facts, resolved subjects, observed/event dates and source IDs |
| Representation | Immutable original turns plus independently retrievable fact keys |
| Evolution | Conservative state chains and semantic attribute histories retaining all observations |
| Retrieval | Lexical/semantic fusion, source-level deduplication, chronological evidence packing |
| Use | Optional request-specific evidence state for updates, arithmetic, time and personalization |

The evidence compiler never sees answers, labels or judge feedback. It records applicable evidence and use constraints; the final model answers from that state and original evidence. It adds one generation call while keeping the final evidence budget fixed. It is a zero-shot adaptation of the relational/request-state distinction in [EvoMemState](https://arxiv.org/html/2611.03743v1), not their RL-trained policy. Other transferred ideas and differences are recorded in `notes/research.json`.

The source-level retrieval variant uses facts as search keys and original turns as values. Each retrieval view contributes at most once per source, duplicate content is removed, and selected sources are presented chronologically with one date header per session. This separates source recovery from memory compression, informed by [JustMem](https://arxiv.org/html/2609.19877v1).

## Results so far

All answerers and current internal judges use `gpt-6-luna`, low reasoning. LongMemEval uses the unchanged official rubric with a Luna judge, so these are not official GPT-4o-judge leaderboard scores.

| LongMemEval-S condition | Development, 24 questions | Validation, 48 disjoint questions |
|---|---:|---:|
| Raw BM25 | 18/24 | 29/48 |
| Facts + attribute histories | 19/24 | 37/48 |
| + request-specific evidence state | 23/24 | 39/48 |

The validation set was frozen before its predictions, with eight questions per task type. Full 500-question indexing and evaluation are running. A stronger BM25 + Qwen3 baseline is also being measured; final selection uses only the 72 development/validation questions, leaving 428 questions for the full test comparison.

On 80 LoCoMo development questions with official captions and 3,000 evidence tokens, raw hybrid retrieval scores 57.15 token F1 and source-level fusion/packing scores 59.23. The full 1,540-question comparison is running. Uniform evidence-state compilation reduced LoCoMo F1 and is not selected for that comparison.

Earlier LoCoMo runs omitted `blip_caption`; they are text-only ablations, retained in `reports/round1.json`. Standard-input results use official automatic captions and exclude image-search `query` hints. `reports/round2.json` contains the completed second-round results. The eight heldout LoCoMo conversations have been evaluated across revisions; report their scores separately from development without treating repeated use as an untouched final test.

## Reproduce

Use Python 3.11 and `requirements.lock.txt` for `.venv`. Embedding scripts use the existing read-only torch environment named below. Set `OPENAI_API_KEY` or use the private local credential file, excluded from version control. `MEMORY_PROXY` defaults to `http://127.0.0.1:41227`.

```bash
.venv/bin/python scripts/build_locomo_many.py 0,1,2,3,4,5,6,7,8,9 --captions
HF_HUB_OFFLINE=1 /home/wqshen/anaconda3-new/envs/user-memory-study/bin/python src/embed.py --convs 0,1,2,3,4,5,6,7,8,9 --captions
.venv/bin/python src/experiment.py evaluate --convs 0,1,2,3,4,5,6,7,8,9 --limit 0 --methods hybrid,hybrid_packed --reader compact --captions --run locomo_full_packed_v2

.venv/bin/python src/longmem.py build --split full --workers 48 --run longmem_full_v2
.venv/bin/python src/consolidate.py
.venv/bin/python src/longmem.py evaluate --split full --methods raw --run longmem_raw_full_v2
.venv/bin/python src/longmem.py evaluate --split full --methods consolidated --state-reader --run longmem_state_full_v2
```

For the stronger LongMemEval baseline, first run `src/embed_longmem.py` with three disjoint `--shard` values, `--shards 3` and explicit CUDA devices. Then use `--methods hybrid`; the dense candidate uses `--methods consolidated_hybrid --state-reader`. Full candidate selection is recorded in `notes/full_protocol_v2.json`.

LoCoMo uses unchanged official Porter-stemmed F1 functions, categories 1-4, comma-split multi-hop answers and the first reference before a semicolon for category 3. Report micro F1, category F1 and macro F1 together. Category 5 is excluded from the 1,540-question comparison. Compound evidence annotations are normalized to source IDs for recall only; answer scoring is unchanged.

LongMemEval uses cleaned S histories rather than oracle histories. `has_answer`, answer-session labels and original IDs are excluded from memory inputs. Local position IDs replace upstream IDs that can contain answer/noans prefixes. Gold evidence is accessed only after retrieval to calculate recall.

Run configs, predictions, API outputs and usage are in `runs/`; successful calls and extracted sessions are cached for resumption. A run-config mismatch requires a new run name. Model, dataset, scorer, prompt and memory fingerprints support reproduction. Default switches preserve the original variants.
