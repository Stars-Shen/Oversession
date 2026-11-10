# A focused next memory hypothesis, not an implemented result

The broad review suggests a dependency-aware temporal memory rather than another
global retrieval-confidence weight. This is a proposed research direction; none
of the formulas below should be described as measured improvements.

## Representation

Keep immutable source observations `o = (entity, attribute, value, event_time,
mention_time, status, source_id)`. State identity includes scope; a preference is
not a mutually exclusive state. Preserve uncertain intervals rather than forcing
exact dates. Original text remains the authority.

A derived profile has a provenance expression. For example,
`(o1 AND o2) OR (o3 AND o4)` distinguishes jointly needed observations from
alternative complete supports. Flat source sets erase that distinction. This
adapts positive database provenance and ATMS labels, without claiming that the
text-to-symbol front end is logically sound by construction.

## Evolution

For each canonical state attribute, maintain an independent run-length posterior:

`growth[r+1] = posterior[r] * predictive(observation | segment[r]) * (1-hazard[r])`

`reset[0] = sum_r posterior[r] * predictive(observation | segment[r]) * hazard[r]`.

The crucial research problem is the predictive likelihood. A sentence paraphrase
is not independent evidence. Observations from the same episode should be grouped;
claims of calibrated posterior truth probabilities require separate calibration.
Do not multiply a second-hand repetition as a fresh independent witness.

When a canonical state changes, update only profiles whose support environments
use that state at the affected event time. Keep older observations for historical
queries, and preserve unaffected support alternatives. Event histories and stable
preferences do not inherit the same invalidation rule.

## One decisive experiment

Freeze the stronger SiliconFlow reading configuration established in this phase.
Test only the dependency-aware evolution mechanism on real knowledge-update and
temporal questions from cleaned LongMemEval-S, retaining the existing original
ledger and source retrieval. Compare against the same reader and memory with the
mechanism disabled. Select a small development cohort and a separate confirmation
cohort before looking at answers. Use the official rubric, and identify any
substitute judge explicitly if OpenAI judging remains paused.

Success requires both an answer-metric improvement and an interpretable distinction:
fewer stale derived conclusions after a real change, without invalidating unrelated
preferences or correct historical answers. Stop if canonicalization/likelihood errors
erase the benefit; do not sweep thresholds to fill an ablation table.

The user asked to seek broad ideas, not to implement every paper. This hypothesis
is deliberately one next mechanism. It is context for a future explicit task,
not authorization to launch another experiment automatically.
