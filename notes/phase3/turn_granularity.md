# Turn-triggered writing versus session extraction

User asks whether a PreCoMem-like turn-level path has been considered. One
[fixed causal turn-anchored backfill experiment](turn_writer.md) now completes,
at 63.5546 F1 versus session writing's 65.4549 on the fixed 80 development
questions. Stop it without a window/prompt sweep; this is not a reproduction
of PreCoMem's online confidence lifecycle.
Do not describe the baseline implementation as turn-level: extract_session sends
one whole dated session and returns atomic facts with one or more turn sources.

Primary check: [PreCoMem v1, sections 2.2 and B.4](https://arxiv.org/html/2610.29467v1).
Its turn t includes user input, response and a rolling working-memory window;
retrieved prior beliefs participate in its read/write loop. Thus a per-turn
update trigger does not imply context-free extraction of an isolated utterance.
Its topic table is a retrieval prior, not the input segmentation unit. Its
turn-level gate is distinct from a per-claim update route (appendix C.4).

For this project the plausible transfer is: incoming turn + bounded preceding
context + retrieved prior memory -> source-grounded atomic observations ->
local relation/belief updates. The current turn must supply new evidence; old
context supplies reference resolution, not duplicate new observations. Assistant
suggestions, other speakers' facts, hypothetical language, and actual user
events need distinct attribution. Keep raw turns, session dates and topic-level
organization above the observations; neither turns nor topics need become the
only retrieval unit.

Potential advantage: finer attribution/order and less within-session omission,
especially local corrections. Risks: repeated writes, fragmented events,
unresolved long-range references, and more calls. None of these establish the
final metric direction without measurement. Large-session compression can lose
information that no downstream relation builder can recover.

The initial relation experiment freezes extraction solely for causal attribution,
not because session extraction is presumed optimal. It now completes and fails:
65.4549 -> 63.9217 F1 on the fixed 80. Stop it; do not carry its derived summaries
into a turn-granularity comparison. A turn-level writer is a separate
main-variable experiment; do not silently combine both changes and attribute a
gain to one. Preserve the session baseline and use the same Luna reader,
source budget and official scorer when that experiment runs.

Two development dialogues contain 788 turns, of which 491 are referenced by
the original 490 facts. Among annotated development evidence, however, 111/118
source references already have a fact pointer; five are missing both from fact
roots and baseline context, across four questions. These are pointer-level
diagnostics, not a semantic completeness audit. The decision-relevant question
for a turn writer is preservation/attribution of useful observations and local
updates, not maximizing fact counts or rewriting all unreferenced greetings.
