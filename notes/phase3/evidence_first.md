# Evidence-first local reading study

The native DeepSeek thinking comparison showed a materially larger gain than
the local CAD or deterministic calendar aids. This experiment asks whether an
explicit, compact working state can improve the available local nonthinking
reader. It adapts claim provenance and question intent from the CoMMA reading,
without reproducing its conflict clustering or four-way answer controller.

One generation emits source-bound observations, the requested operation, one
short scope/calculation check, and the final answer. It sees the same v2 source
context and question as the local plain baseline. The original answer-format
instructions apply inside the final answer field; no gold, category, reference,
or prior answer is supplied. The generation cap increases from 192 to 384 tokens.
This changes reading compute, not memory content or the 3000-token evidence budget.

Pilot scope: the first 24 questions in the already frozen 80-question development
manifest, in its original order. The 24 plain outputs are reused. Before scoring,
the go/no-go rule is fixed: expand to 80 only if paired F1 improves by at least
3 points, no output is truncated, and at least 95% of outputs have valid JSON and
source membership. This is a development triage rule, not a significance claim.
No prompt or budget grid is planned.

The parser accepts a single JSON object, optionally fenced. Every observation
must cite an input source and contain a nonempty fact; a factual answer needs an
observation. An invalid structured output deterministically becomes an abstention
and remains in the score. Source membership does not establish semantic entailment.

Artifacts: `runs/phase3/local_evidence_first_pilot/` and, after scoring,
`reports/local_evidence_first_pilot.json`. Entry points:

```bash
.venv/bin/python scripts/evidence_first_study.py prepare
# Use the same existing torch environment and UUID-bound cards 1-3:
# .../bin/python src/local_cad.py generate --run local_evidence_first_pilot
.venv/bin/python scripts/evidence_first_study.py score
```

The paid API queue remains stopped after HTTP 402. This local result must not be
mixed into the missing DeepSeek full-evaluation rows. A useful development gain
must still be checked on separate conversations before claiming broad improvement.

## Pilot result and one interface correction

The 24-question strict pilot failed its preregistered gate: plain F1 46.8281,
evidence-first 45.5683 (−1.2598 points), 7 wins / 3 losses / 14 ties. Two outputs
cited IDs outside the input (one fabricated turn ID, one session header). Both
became abstentions under the stated policy; no truncation occurred. A separate
diagnostic that scores the raw answer field gives +4.8513 points, but it does not
pass the provenance contract and is not reported as an achieved guarded gain.

One corrective version accepts a valid initial output and makes exactly one
constrained retry when the source/format contract fails. The retry's JSON source-ID
fields are enums of actual input sources. The evidence-first prompt is unchanged;
no answer-specific fixes are made. The 22 valid pilot outputs are reused literally
after verifying identical inputs. Two invalid pilot attempts need only the repair
call; the other 56 questions require new first attempts. All inference compute,
including failed first attempts, remains recorded.

An attempt to use the existing read-only vLLM 0.10.1.1+cu118 environment failed
during initialization: its MoE library has an unresolved `shuffle_rows` symbol.
No benchmark predictions came from that backend, and no shared environment was
changed. The experiment therefore retains the original HF/torch backend and its
cached plain baseline. XGrammar is imported from the existing dependency directory
after the original torch/transformers imports; its source-enum rejection and GPU
logit mask were checked directly. This provides the source constraint without
modifying installed libraries or pursuing an inference-engine rebuild.

The full 80-question development scope is fixed. A gain of at least 3 F1 points
with no final truncation is required before expanding beyond development. No
further prompt/source-format grid is planned. Artifacts:
`runs/phase3/local_evidence_first_guarded_dev/`. Preparation:
`scripts/prepare_guarded_evidence.py`; generation: `src/local_cad.py generate
--run local_evidence_first_guarded_dev`; scoring: `scripts/evidence_first_study.py
score --run local_evidence_first_guarded_dev`. A valid source enum establishes
membership only, not that a paraphrased observation entails the final answer.

One inspected failure shows why this distinction matters: the model described a
park visit while the cited source was a generic reply plus an automatic park-image
caption. Another confused the source date and even computed "next month" incorrectly.
This motivates distinguishing immutable source fields from model-generated claims
in any subsequent reader design; valid addresses alone are insufficient.
`reports/evidence_binding_diagnostic.json` records the inspected source text and
metadata without consulting gold answers. In particular, the source clock for the
talent-show statement is 28 August 2023, so "next month" refers to September at
month precision. Neither a fabricated day nor October follows from that source.

If the corrected full development result is still weak, stop formatting changes.
The next substantive representation hypothesis is to materialize source fields
externally rather than asking the model to regenerate them: select source roots,
read their actual timestamps and human/caption channels from the immutable ledger,
and execute unambiguous calendar operations before final answering. This is a
prospective design, not a measured improvement or authorization to restart old jobs.

## Corrected development result: stop

All 80 questions completed: plain F1 **58.4042**, guarded evidence-first
**58.5071** (+0.1029 points), 17 wins / 12 losses / 51 ties. Ten questions used
the single constrained repair; all final outputs pass source membership and none
is truncated. Total completion tokens, including failed first attempts, are 9111.
The gain fails the fixed 3-point expansion threshold. Do not expand this protocol
or try another prompt/grammar variant. A valid source address does not repair a
misread event, person, or timestamp. The report is
`reports/local_evidence_first_guarded_dev.json`.
