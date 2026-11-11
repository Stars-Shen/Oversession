# Versioned Evidence Memory

A five-stage long-dialog memory framework with completed LoCoMo and LongMemEval-S comparisons against the same-backbone hybrid retrieval baseline. Full results and scope checks are in `reports/final_v2.json`; these comparisons do not establish a global SOTA rank.

| Stage | Implementation |
|---|---|
| Extraction | Question-blind atomic facts, resolved subjects, observed/event dates and source IDs |
| Representation | Immutable original turns plus independently retrievable fact keys |
| Evolution | Conservative state chains and semantic attribute histories retaining all observations |
| Retrieval | Lexical/semantic fusion, source-level deduplication, chronological evidence packing |
| Use | Optional request-specific evidence state for updates, arithmetic, time and personalization |

The evidence compiler never sees answers, labels or judge feedback. It records applicable evidence and use constraints; the final model answers from that state and original evidence. It adds one generation call while keeping the final evidence budget fixed. It is a zero-shot adaptation of the relational/request-state distinction in [EvoMemState](https://arxiv.org/html/2611.03743v1), not their RL-trained policy. Other transferred ideas and differences are recorded in `notes/research.json`.

The source-level retrieval variant uses facts as search keys and original turns as values. Each retrieval view contributes at most once per source, duplicate content is removed, and selected sources are presented chronologically with one date header per session. This separates source recovery from memory compression, informed by [JustMem](https://arxiv.org/html/2609.19877v1).

## Benchmark results

| Benchmark | Hybrid retrieval baseline | VEM | Improvement |
|---|---:|---:|---:|
| LoCoMo, 1,540 nonadversarial questions, micro F1 | 58.71 | **61.78** | **+3.07 points** |
| LongMemEval-S, 500 questions, official judge accuracy | 76.00% | **80.40%** | **+4.40 points** |

All final readers use `gpt-6-luna` with low reasoning. Final evidence budgets are 3,000 tokens for LoCoMo and 6,000 for LongMemEval, counted with `cl100k_base`. The evidence compiler adds one model call; total inference compute is not claimed to be matched.

On the 428 LongMemEval questions excluded from development and validation, official accuracy rises from **76.40% to 79.91% (+3.50 points)**. All 500 original memory ledgers retain their original turns and facts through consolidation. The full index contains 153,656 fact entries and 23,298 usable attribute histories. Three of 23,867 session extractions failed; original evidence remained available for retrieval.

Single-session-user accuracy falls from 69/70 to 64/70. Knowledge updates improve from 66/78 to 73/78, and preference use from 10/30 to 19/30; all task-type results are retained in the final report.

### Development trace

The following development/validation trace uses a Luna substitute judge with unchanged official rubrics. The full scores above use the official `gpt-4o-2024-08-06` snapshot, rubric, temperature 0 and 10-token judge output setting.

| LongMemEval-S condition | Development, 24 questions | Validation, 48 disjoint questions |
|---|---:|---:|
| Raw BM25 | 18/24 | 29/48 |
| BM25 + Qwen3 hybrid | — | 35/48 |
| Facts + attribute histories | 19/24 | 37/48 |
| + request-specific evidence state | 23/24 | 39/48 |

The validation set was frozen before its predictions, with eight questions per task type. The sparse track/state profile was selected using only the 72 development/validation questions. Dense fusion into the track pipeline did not improve validation and was not expanded. Complete official baselines are 70.8% for raw BM25 and 76.0% for hybrid retrieval; the selected candidate achieves 80.4%.

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

To regenerate the completed scope, provenance and score comparison from saved artifacts, run `.venv/bin/python scripts/report_full.py`. It requires exact coverage of all 500 LongMemEval and 1,540 LoCoMo questions, checks actual response models and official judgments, verifies evidence budgets, and compares the original and consolidated memory ledgers.

## SiliconFlow research phase: full evaluation incomplete

Broader reading of PreCoMem, RGMem, HippoRAG 2, A-Mem, Zep, database provenance,
truth maintenance and Bayesian change detection is recorded in
[phase-3 research notes](notes/phase3/research.md). Several structural experiments
were rejected; the cross-encoder's small development gain failed separate validation.

With the same frozen v2 memory, evidence, prompts and 3000-token final context,
`deepseek-ai/DeepSeek-V4.1-Flash` native thinking produced:

| LoCoMo cohort | Thinking disabled | Thinking enabled | Difference |
|---|---:|---:|---:|
| Category-balanced separate validation, 160 questions | 59.68 | 65.40 | +5.72 F1 points |
| Available confirmation pairs excluding development/validation, 603 questions | 60.04 | 63.94 | +3.90 F1 points |

The confirmation paired bootstrap interval is [+1.59, +6.33] points. Single-hop
performance declines in that cohort; this is not a uniform improvement. Thinking
uses extra inference compute and a larger completion limit, and the writer remains
the frozen Luna writer. This is a reader-compute result, not a new memory algorithm.

The interrupted confirmation cohort underrepresents single-hop questions, where
thinking declines. A [category-mixture audit](notes/phase3/category_mix.md) shows
that transferring its category effects to the still-eligible confirmation pool's
proportions would give only +0.93 points. This is a sensitivity calculation,
not a completed score: within-category completion may also be nonrandom.
The measured +3.90 cohort gain must not be presented as a full-benchmark gain.

The full run stopped on SiliconFlow HTTP 402 (insufficient balance), with 1855/3080
outputs and 924 complete pairs saved. **There is no completed 1540-question phase-3
score.** No new OpenAI calls were made. A balance guard now stops the queued run on
the first 402. All caches support resumption with the same configuration after funding.

See [scope-audited report](reports/phase3.json) and
[SiliconFlow reproduction instructions](notes/phase3/reproduce.md). The earlier
completed v2 benchmark results above are preserved separately.

Local follow-up studies use Qwen3-30B-A3B-Instruct-2507 with the same frozen writer,
source retrieval and 3000-token evidence budget. Source-bound structured reading
with one constrained repair changes development F1 only 58.40 → 58.51. Uniform
canonical fact values reduce it to 56.93. Selecting canonical values only for
time questions gives 62.52 in development, but just **51.58 → 53.05** on the
160-question separate-conversation cohort (paired question-bootstrap interval
for the gain: **[−1.54, +4.50] points**). These weak candidates are not expanded
or selected as breakthroughs. No new paid calls were made. See
[research decisions](notes/phase3/decisions.md),
[value-layer protocol](notes/phase3/canonical_values.md), and
[completed local comparison](reports/local_selective_canonical_validation.json).

A matched Qwen3-4B native-thinking trial improves its own development control
52.44 → 58.25 F1, but remains below the existing 30B reader's 58.40, so it is not
expanded. One fixed low-rank reader adaptation, trained only on conversation 0,
improves conversation 1 by 2.85 points but falls **46.25 → 45.37** on the
160-question separate-conversation check, versus 51.58 for the existing 30B reader.
That adapter is rejected. See the [native-capacity protocol](notes/phase3/local_native_capacity.md)
and [learned-reader results](notes/phase3/learned_reader_hypothesis.md).

One fixed [30B source-supervised reader study](notes/phase3/source_supervised_reader.md),
trained on conversation 0's 131 fully supported questions, improves the other
dialogue's 81-question validation **58.62 -> 61.20 F1 (+2.58)**. This misses the
original +3-point triage gate, which remains false. One explicit exception checks
the unchanged final adapter on 160 questions across eight conversations:
**51.58 -> 55.70 (+4.12)**, with a question-bootstrap interval of [-0.88, +9.23]
points and zero capped outputs. Target-mixture sensitivity is +4.29 points;
multi-hop declines by 3.97 points. A 1307-question paired comparison was started,
but the queued adapter expansion is cancelled after the stronger-reader check
below. The original control completes at **57.70 F1**, versus the same-cohort
old Luna reference's 61.42; all cached answers and the adapter are preserved.
There is no completed 1307-question paired adapter score.

The stronger same-context DeepSeek deliberate reference on those 160 questions
is **65.40 F1**, 9.70 points above the adapted local reader. A gain against the
weaker local control therefore does not establish the broader research goal.
One [full-history development test](notes/phase3/full_history.md) now examines
whether the 3K information budget is the bottleneck, with the original unadapted
reader and causal prefix reuse. It completes at **50.97 F1**, below the same
reader's 3K retrieval score of 58.40 and the strongest development reader's
65.57. This candidate is stopped without an ordering or length sweep.

A fixed [same-size native reader replacement](notes/phase3/native_30b_reader.md)
now assesses Qwen3-30B-A3B-Thinking-2507 using the exact original 3K evidence.
Its pinned weights are downloaded and verified; the fixed 80-question generation
is running on two GPU replicas. F1 results are pending.
This changes reader weights and inference policy, not the memory algorithm.
