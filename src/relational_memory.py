"""Question-blind incremental relations with immutable, complete support groups."""
from collections import defaultdict
import json
import re
from nltk.stem import PorterStemmer
from api import atomic_json, digest
from memory import BM25, tokens

PROMPT = '''Maintain a relational memory BEFORE future questions are known.
Input contains past atomic facts, incoming atomic facts and active derived cards.
Return JSON {"cards":[{"subject":"person/entity", "relation":"specific relation",
"scope":"specific object/event distinguishing this relation", "text":"precise self-contained relation",
"support_groups":[["fact ID", "fact ID"]], "revises":null}]}.
Create at most 8 cards, only if connecting facts from DIFFERENT sessions adds a
specific relation that no individual fact states. Examples of operations: resolve
an explicit reference to an earlier named object; bind a later report to an earlier
plan when identity is supported; connect a stated cause to its later consequence;
consolidate an explicitly recurring concrete relationship or dated event sequence.
Do not assume a planned event happened. Do not invent identity between merely
similar events. Do not derive personality traits or broad topic summaries. Do not
create a card just because facts share a hobby, person or vocabulary. An empty
list is valid. Do not repeat an existing card unless new evidence changes it.
Each support group must contain 2-4 supplied fact IDs, span at least two sessions,
include an incoming fact, and JOINTLY support the ENTIRE card text. Alternative
groups must each independently support the full text. Preserve exact names,
dates, uncertainty, historical facts and distinctions between planned/actual.
If incoming evidence refines/corrects an active card, copy its subject/relation/
scope EXACTLY and put its ID in revises. Keep historical qualifications in text.
Revisions are local to that key; never erase unrelated knowledge. Otherwise use
revises:null and a new specific key. Original facts are authoritative.'''


def key(card):
    return tuple(re.sub(r'\W+', ' ', card[k].lower()).strip()
                 for k in ('subject', 'relation', 'scope'))


def active_cards(cards):
    replaced = {c['revises'] for c in cards if c.get('revises')}
    return [c for c in cards if c['id'] not in replaced]


def accept_cards(proposals, memory, session, prior):
    """Check provenance structure and local versioning, not semantic entailment."""
    facts = {f['id']: f for f in memory['facts']}
    current = {f['id'] for f in memory['facts'] if f['session'] == session['id']}
    order = next(i for i, s in enumerate(memory['sessions']) if s['id'] == session['id'])
    known_sessions = {s['id'] for s in memory['sessions'][:order + 1]}
    active = {c['id']: c for c in active_cards(prior)}
    accepted, rejected = [], []
    for proposal in proposals[:8]:
        reason = None
        if not all(isinstance(proposal.get(k), str) and proposal[k].strip()
                   for k in ('subject', 'relation', 'scope', 'text')):
            reason = 'missing_text_or_key'
        groups = proposal.get('support_groups')
        if not isinstance(groups, list) or not groups:
            reason = 'missing_support'
        else:
            for group in groups:
                if not isinstance(group, list) or not all(isinstance(x, str) for x in group):
                    reason = 'invalid_group'; break
                if not 2 <= len(group) <= 4 or len(set(group)) != len(group):
                    reason = 'invalid_group_size'; break
                if not set(group) <= facts.keys():
                    reason = 'unknown_fact'; break
                sessions = {facts[fid]['session'] for fid in group}
                if not sessions <= known_sessions:
                    reason = 'future_fact'; break
                if len(sessions) < 2 or not set(group) & current:
                    reason = 'not_cross_session_increment'; break
        if reason is None:
            revises = proposal.get('revises')
            matches = [c for c in active.values() if key(c) == key(proposal)]
            if revises is not None and (not isinstance(revises, str) or revises not in active
                                       or key(active[revises]) != key(proposal)):
                reason = 'nonlocal_revision'
            elif revises is None and matches:
                reason = 'duplicate_key_without_revision'
        if reason:
            rejected.append(dict(proposal=proposal, reason=reason)); continue
        card = {k: proposal[k].strip() for k in ('subject', 'relation', 'scope', 'text')}
        card.update(support_groups=groups, revises=revises, session=session['id'],
                    observed_at=session['date'], order=order)
        card['id'] = 'r_' + digest(card)[:16]
        accepted.append(card)
        if revises is not None:
            del active[revises]
        active[card['id']] = card
    return accepted, rejected


def build_ledger(memory, api, path):
    signature = digest(dict(memory=memory, prompt=PROMPT, model=api.model))
    if path.exists():
        ledger = json.loads(path.read_text()); assert ledger['signature'] == signature
    else:
        ledger = dict(signature=signature, cards=[], steps=[])
    for i, session in enumerate(memory['sessions']):
        if i < len(ledger['steps']):
            continue
        past_sessions = {s['id'] for s in memory['sessions'][:i]}
        def facts_for(sessions):
            return [{k: f.get(k) for k in ('id', 'subject', 'slot', 'kind', 'text',
                    'session', 'observed_at', 'event_date')}
                    for f in memory['facts'] if f['session'] in sessions]
        incoming = facts_for({session['id']}); past = facts_for(past_sessions)
        step = dict(session=session['id'], accepted=0, rejected=[], new_paid_call=False,
                    estimated_new_cost_usd=0.)
        if past and incoming:
            payload = dict(incoming=incoming, past=past, active_cards=active_cards(ledger['cards']))
            result = api.call(PROMPT, json.dumps(payload, ensure_ascii=False),
                              max_tokens=4096, json_output=True, effort='low')
            if not result.get('complete', True):
                raise RuntimeError('Incomplete relation build; no automatic repair or retry')
            if result['new_paid_call'] and result.get('served_service_tier') != 'flex':
                raise RuntimeError('Relation build was not confirmed Flex; stop')
            accepted, rejected = accept_cards(result['parsed']['cards'], memory, session, ledger['cards'])
            ledger['cards'].extend(accepted)
            step.update(accepted=len(accepted), rejected=rejected,
                        request_hash=result['request_hash'], usage=result['usage'],
                        new_paid_call=result['new_paid_call'],
                        estimated_new_cost_usd=result['estimated_new_cost_usd'])
        ledger['steps'].append(step); atomic_json(path, ledger)
        print(path.stem, session['id'], 'new relations', step['accepted'],
              'active', len(active_cards(ledger['cards'])), flush=True)
    return ledger


class RelationMemory:
    def __init__(self, memory, cards):
        self.memory = memory
        self.cards = active_cards(cards)
        self.stemmer = PorterStemmer()
        self.words = lambda text: [self.stemmer.stem(w) for w in tokens(text)]
        self.index = BM25([' '.join(self.words(c['text'])) for c in self.cards])
        self.names = set(self.words(' '.join(f.get('subject', '') for f in memory.facts)))

    def render(self, selected, bundles):
        m = self.memory; parts = []
        if bundles:
            parts.append('Derived cross-session relations (original turns are authoritative):')
            for card, group, roots in bundles:
                parts.append('[{}; observed {}; sources {}] {}'.format(
                    card['id'], card['observed_at'], ', '.join(sorted(roots, key=m.turn_pos.get)), card['text']))
        sid = None
        for mid in sorted(selected, key=m.turn_pos.get):
            t = m.by_turn[mid]
            if t['session'] != sid:
                sid = t['session']; parts.append('Session {} | {}'.format(sid, t['date']))
            parts.append('[{}] {}: {}'.format(mid, t['speaker'], t['text']))
        return '\n'.join(parts)

    def retrieve(self, question, budget, encoder, dense, enabled=True):
        m = self.memory; baseline = m.retrieve_packed(question, budget, encoder, dense)
        if not enabled or not self.cards:
            return baseline
        scores = defaultdict(float)
        for ranking, is_fact in [(m.raw_index.rank(question), False), (dense['raw'], False),
                                 (m.fact_index.rank(question), True), (dense['facts'], True)]:
            seen = set()
            for rank, (i, _) in enumerate(ranking):
                for mid in m.facts[i]['sources'] if is_fact else [m.turns[i]['id']]:
                    if mid not in seen:
                        scores[mid] += 1 / (40 + rank); seen.add(mid)
        content = set(self.words(question)) - self.names
        selected, bundles = set(), []
        for ci, _ in self.index.rank(' '.join(self.words(question))):
            card = self.cards[ci]
            if not content & set(self.words(card['text'])):
                continue
            choices = []
            for group in card['support_groups']:
                roots = {mid for fid in group for mid in m.by_fact[fid]['sources']}
                if roots & set(baseline['source_ids']):
                    score = sum(scores[mid] for mid in roots) / len(roots)
                    choices.append((score, group, roots))
            if not choices:
                continue
            _, group, roots = sorted(choices, key=lambda x: (-x[0], tuple(x[1])))[0]
            candidate = (card, group, roots)
            text = self.render(selected | roots, bundles + [candidate])
            if len(encoder.encode(text)) <= budget:
                selected.update(roots); bundles.append(candidate)
            if len(bundles) == 2:
                break
        if not bundles:
            return baseline
        for mid in sorted(scores, key=lambda mid: (-scores[mid], m.turn_pos[mid])):
            if mid not in selected and len(encoder.encode(self.render(selected | {mid}, bundles))) <= budget:
                selected.add(mid)
        text = self.render(selected, bundles)
        return dict(text=text, tokens=len(encoder.encode(text)), source_ids=sorted(selected, key=m.turn_pos.get),
                    facts=[], relations=[dict(id=c['id'], support_group=g,
                    sources=sorted(roots, key=m.turn_pos.get)) for c, g, roots in bundles])
