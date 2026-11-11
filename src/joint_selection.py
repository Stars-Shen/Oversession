"""Single-pass source lookup through a frozen global fact/residual index."""
from api import digest

PROMPT = '''Select the original evidence needed to answer a question about dated conversation memory.
Return JSON {"ids":["F0","R0"]}, using ONLY supplied entry aliases. Do not answer
the question, generate observations, or explain the selection. Facts are an index
to original source turns; residual entries preserve turns not indexed by facts.
Choose a small JOINTLY sufficient set, not merely similar topics. Match the exact
person, relation, object and time scope. For common/shared items, include witnesses
for EACH person. For lists/counts, include ALL contributing items/events, not just
the strongest example; distinguish repeated mentions, plans and actual events.
For temporal questions, include original event/observation clocks and necessary
anchors. Preserve historical observations when relevant. For explicitly requested
inferences, select relevant experiences, preferences and constraints; an exact
verbatim answer is not required. Include earlier evidence needed to resolve a
later reference. Avoid unrelated encouragement, themes, friendships or dates.
Use no outside personal facts. The selected originals will be read by a separate
answerer under a 3000-token maximum. Do not pad the selection with weak matches.
Return an empty list if no relevant memory exists.'''


def memory_index(memory):
    entries, roots = [], {}
    for i, fact in enumerate(memory.facts):
        alias = 'F' + str(i)
        entries.append(dict(id=alias, **{k: fact.get(k) for k in
            ('subject', 'slot', 'observed_at', 'event_date', 'text')}))
        roots[alias] = list(fact['sources'])
    covered = {mid for refs in roots.values() for mid in refs}
    for i, turn in enumerate(t for t in memory.turns if t['id'] not in covered):
        alias = 'R' + str(i)
        entries.append(dict(id=alias, **{k: turn[k] for k in ('speaker', 'date', 'text')}))
        roots[alias] = [turn['id']]
    return dict(entries=entries, roots=roots, signature=digest(entries))


def selected_context(memory, index, selection, budget, encoder):
    """Return None for an unusable WHOLE selection; callers reuse exact baseline."""
    aliases = selection.get('ids') if isinstance(selection, dict) else None
    if not isinstance(aliases, list) or not aliases or not all(isinstance(a, str) for a in aliases):
        return None
    if not set(aliases) <= index['roots'].keys():
        return None
    roots = {mid for alias in aliases for mid in index['roots'][alias]}
    selected = set(roots)
    for mid in roots:
        pos = memory.turn_pos[mid]
        if pos and memory.turns[pos - 1]['session'] == memory.by_turn[mid]['session']:
            selected.add(memory.turns[pos - 1]['id'])
    parts, sid = [], None
    for mid in sorted(selected, key=memory.turn_pos.get):
        turn = memory.by_turn[mid]
        if turn['session'] != sid:
            sid = turn['session']; parts.append('Session {} | {}'.format(sid, turn['date']))
        parts.append('[{}] {}: {}'.format(mid, turn['speaker'], turn['text']))
    text = '\n'.join(parts); size = len(encoder.encode(text))
    if size > budget:
        return None
    return dict(text=text, source_ids=sorted(selected, key=memory.turn_pos.get), tokens=size,
                facts=[], selected_aliases=list(dict.fromkeys(aliases)), roots=sorted(roots, key=memory.turn_pos.get))
