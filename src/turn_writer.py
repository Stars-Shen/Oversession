"""Causal target-turn extraction with a frozen completed-session history view."""
from api import digest
from memory import BM25, EXTRACT, evolve

PROMPT = EXTRACT + '''
This request has ONE target turn, preceding context, and historical facts with
their source turns. Extract ONLY new observations supplied by the target turn.
Use context/history to resolve references and attribute people and dates, not
to repeat information that appears only in context. Every output fact must cite
the target turn ID, and must also cite contextual turns needed to resolve its
meaning. All citations must be IDs of visible raw turns, never fact IDs. A target
can yield zero, one or several facts. Questions, suggestions and generic support
do not establish that the mentioned personal event or attribute is true. Preserve
uncertainty and distinguish plans, actual events and other people's experiences.
Reuse a historical subject/slot/scope for the same attribute where appropriate,
but keep distinct events and people distinct. Return the same JSON fact schema.'''


def inputs(data):
    """No QA fields or future observations cross this allowlist."""
    turns = [t for s in data['sessions'] for t in s['turns']]
    by_turn = {t['id']: t for t in turns}; position = {t['id']: i for i, t in enumerate(turns)}
    session_pos = {s['id']: i for i, s in enumerate(data['sessions'])}
    indices = {}
    for sid, si in session_pos.items():
        facts = [f for f in data['facts'] if session_pos[f['session']] < si]
        indices[sid] = facts, BM25([f.get('subject', '') + ' ' + f.get('slot', '') + ' ' + f['text'] for f in facts])
    def raw(t):
        return {k: t[k] for k in ('id', 'speaker', 'text', 'date', 'session')}
    out = []
    for pos, target in enumerate(turns):
        context = turns[max(0, pos - 8):pos]
        facts, index = indices[target['session']]
        query = target['text'] + ' ' + ' '.join(t['text'] for t in context)
        history = [facts[i] for i, _ in index.rank(query, 8)]
        refs = {mid for f in history for mid in f['sources']}
        assert all(position[mid] < pos for mid in refs)
        payload = dict(target=raw(target), preceding=[raw(t) for t in context],
                       historical_facts=[{k: f.get(k) for k in ('id', 'subject', 'slot', 'scope',
                           'text', 'kind', 'event_date', 'observed_at', 'sources')} for f in history],
                       historical_sources=[raw(by_turn[mid]) for mid in sorted(refs, key=position.get)])
        out.append(dict(target_id=target['id'], order=pos, payload=payload))
    return out


def observations(row, proposals):
    target = row['payload']['target']
    visible = {t['id'] for t in [target] + row['payload']['preceding'] + row['payload']['historical_sources']}
    facts, rejected = [], []
    for proposal in proposals:
        reason = None
        refs = proposal.get('sources')
        if not isinstance(refs, list) or not all(isinstance(mid, str) for mid in refs):
            reason = 'invalid_sources'
        elif target['id'] not in refs:
            reason = 'no_target_evidence'
        elif not set(refs) <= visible:
            reason = 'invisible_source'
        elif not all(isinstance(proposal.get(k), str) and proposal[k].strip()
                     for k in ('subject', 'slot', 'text', 'kind')):
            reason = 'invalid_fact_fields'
        elif proposal['kind'] not in ('event', 'state', 'preference', 'plan'):
            reason = 'invalid_kind'
        if reason:
            rejected.append(dict(proposal=proposal, reason=reason)); continue
        fact = {k: proposal.get(k) for k in ('subject', 'slot', 'scope', 'text', 'kind', 'single', 'event_date')}
        fact.update(sources=list(dict.fromkeys(refs)), session=target['session'], observed_at=target['date'],
                    order=row['order'], origin_turn=target['id'])
        fact['id'] = 'tf_' + digest(fact)[:16]
        facts.append(fact)
    # Identical output objects within one turn are a duplicate, not new evidence.
    return list({f['id']: f for f in facts}.values()), rejected


def assemble(data, rows):
    facts = [f for row in sorted(rows, key=lambda r: r['order']) for f in row['facts']]
    return dict(schema='turn-observations-v1', sessions=data['sessions'], facts=evolve(facts),
                writer_prompt_hash=digest(PROMPT), historical_context='frozen earlier-session facts',
                extraction_incompletes=[r['target_id'] for r in rows if not r['complete']])
