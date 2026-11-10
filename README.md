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
| BM25 + Qwen3 hybrid | — | 35/48 |
| Facts + attribute histories | 19/24 | 37/48 |
| + request-specific evidence state | 23/24 | 39/48 |

The validation set was frozen before its predictions, with eight questions per task type. Full 500-question indexing and candidate evaluation are running. The selected sparse track/state system leads the stronger hybrid baseline on validation (39/48 vs 35/48); dense fusion into the track pipeline did not improve validation. Selection uses only the 72 development/validation questions, leaving 428 questions for the full test comparison. The completed official GPT-4o-2024-08-06 baselines on all 500 questions are 70.8% for raw BM25 and 76.0% for hybrid retrieval. Candidate official scores remain pending.

On all 1,540 LoCoMo nonadversarial questions with official captions and 3,000 evidence tokens, raw hybrid retrieval scores **58.71 micro F1 / 50.86 macro F1** and source-level fusion/packing scores **61.78 / 53.49**. Evidence source recall rises from 85.21% to 92.74%. On the eight heldout conversations (1,307 questions), micro F1 rises from 58.74 to 61.42. Full results and protocol details are in `reports/locomo_full_v2.json`. Uniform evidence-state compilation reduced LoCoMo F1 and is not selected for this comparison.

Earlier LoCoMo runs omitted `blip_caption`; they are text-only ablations, retained in `reports/round1.json`. Standard-input results use official automatic captions and exclude image-search `query` hints. `reports/round2.json` contains the completed second-round results. The eight heldout LoCoMo conversations have been evaluated across revisions; report their scores separately from development without treating repeated use as an untouched final test.

## Mechanism and interpretation

Each fact carries its original source IDs, subject, attribute, scope, observation time and optional event time. Raw evidence remains immutable. Consolidation groups semantically equivalent attributes into histories; it does not overwrite observations or assert that every later mention invalidates earlier facts. This prototype consolidates at memory checkpoints rather than claiming a constant-time online graph update.

For source-level retrieval, each lexical or semantic view contributes `1 / (40 + best_source_rank)` once per original turn, even if several fact keys point to that turn. The final values are original turns. Shared session date headers and chronological ordering reduce duplicated metadata within the fixed evidence budget.

For the state-aware profile, a request state identifies the target attribute, time scope, operation and the applicability of cited observations. Unknown source IDs are rejected. The compiled state is limited to 1,000 tokens; original evidence is repacked into the remaining reader budget. It is advisory, with original evidence retaining authority. This improves memory use without claiming that provenance checks eliminate semantic extraction errors.

The two evaluated read profiles are explicitly configured. The source-packing profile was selected for LoCoMo, and attribute histories plus the evidence compiler for LongMemEval. An automatic universal profile selector or a trained RL evolution policy has not been implemented or claimed.

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

For overlapping stages during the long index run, `src/materialize.py` converts only fully cached sessions into the same immutable memory format without API calls. Run consolidation over those memories, then evaluate with `--ready-only`. The frozen configuration still contains all 500 IDs; partial summaries state `expected_n=500` and `complete=false`. Resume the same run as more memories become available. Use `src/official_judge.py RUN_NAME` on completed prediction batches to obtain official-snapshot judgments; its cache prevents repeated scoring.
