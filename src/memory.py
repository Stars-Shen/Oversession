"""Versioned Evidence Memory: source ledger, atomic facts, conservative state chains."""
from collections import Counter, defaultdict
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from api import API, ROOT, digest, atomic_json

EXTRACT = '''You build a persistent memory of a conversation BEFORE any future questions are known.
Extract detailed, atomic facts grounded in the supplied turns. Resolve pronouns to the speaker or
named entity. Keep names, quantities, reasons, relationships, locations, emotions, preferences,
plans, experiences, exact dates and distinctive details. Do not invent information from images.
Resolve relative dates only when the session date makes them unambiguous. Keep uncertainty.
Return JSON {"facts": [{"subject": "canonical person/entity", "slot": "short lower_snake_case attribute",
"text": "one self-contained fact including all key details", "kind": "event|state|preference|plan",
"single": false, "scope": "entity or topic qualifying this attribute", "event_date": null,
"sources": ["turn id"]}]}.
Use single=true ONLY for mutually exclusive state attributes such as current residence, current
employer, current relationship status; NEVER for hobbies, preferences, past events, lists or plans.
A state fact reports a current state, not a past anecdote. event_date is ISO YYYY-MM-DD or null.
All sources must be IDs of turns from this session that directly support the fact.
Do not store greetings or generic encouragement. Preserve all substantive facts, not just a summary.'''

STOP = set('a an the is are was were be been being i me my mine we us our you your he she his her it its they them their do does did of to in on at by for from with and or but that this these those what which when where who how why can could would should have has had as about into than then also just really very so some any tell please'.split())


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class BM25:
    def __init__(self, texts):
        self.docs = [Counter(tokens(t)) for t in texts]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg = sum(self.lengths) / max(1, len(texts))
        self.df = Counter(w for c in self.docs for w in c)
        self.postings = defaultdict(list)
        for i, c in enumerate(self.docs):
            for w, f in c.items():
                self.postings[w].append((i, f))

    def rank(self, query, limit=None):
        scores = defaultdict(float)
        for w in set(tokens(query)):
            idf = math.log(1 + (len(self.docs) - self.df[w] + .5) / (self.df[w] + .5))
            for i, f in self.postings[w]:
                scores[i] += idf * f * 2.5 / (f + 1.5 * (.25 + .75 * self.lengths[i] / max(1, self.avg)))
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:limit]


def locomo_sessions(sample, include_captions=False):
    conv = sample['conversation']
    sessions = []
    for key in sorted((k for k in conv if re.fullmatch(r'session_\d+', k)), key=lambda x: int(x.split('_')[1])):
        date = conv[key + '_date_time']
        turns = [dict(id=t['dia_id'], speaker=t['speaker'],
                      text=t['text'] + (' [Shared image caption (automatic): ' + str(t['blip_caption']) + ']'
                                        if include_captions and t.get('blip_caption') else ''), date=date,
                      session=key, order=len(sessions)) for t in conv[key]]
        sessions.append(dict(id=key, date=date, turns=turns))
    return sessions


def extract_session(session, api):
    content = json.dumps(session, ensure_ascii=False)
    path = ROOT / 'runs/fact_cache' / (digest(dict(prompt=EXTRACT, session=session, model=api.model)) + '.json')
    if path.exists():
        return json.loads(path.read_text())
    result = api.call(EXTRACT, content, max_tokens=12000, json_output=True)
    valid = {t['id'] for t in session['turns']}
    facts = []
    rejected = 0
    rejected_kinds = 0
    for f in result['parsed']['facts']:
        refs = f.get('sources', [])
        if not refs or not set(refs) <= valid:
            rejected += 1
            continue
        f['kind'] = {'experience': 'event', 'goal': 'plan'}.get(f['kind'], f['kind'])
        if f['kind'] not in ('event', 'state', 'preference', 'plan'):
            rejected_kinds += 1
            continue
        f = dict(f, id='f_' + digest([session['id'], f])[:16], session=session['id'],
                 observed_at=session['date'], order=session['turns'][0]['order'])
        facts.append(f)
    out = dict(facts=facts, rejected_source_facts=rejected, rejected_kind_facts=rejected_kinds,
               api_hash=result['request_hash'], usage=result['usage'])
    atomic_json(path, out)
    return out


def evolve(facts):
    """Append-only version links. Events and multivalued preferences never invalidate."""
    chains = defaultdict(list)
    out = []
    for f in sorted(facts, key=lambda x: (x['order'], x['id'])):
        f = dict(f, previous=[], next=[])
        if f.get('single') and f['kind'] == 'state':
            key = tuple(re.sub(r'\W+', ' ', str(f.get(k, '')).lower()).strip()
                        for k in ('subject', 'slot', 'scope'))
            chain = chains[key]
            # Same-session facts remain parallel; there is no within-session temporal claim.
            prior = [old for old in chain if old['order'] < f['order']]
            if prior:
                latest_order = max(old['order'] for old in prior)
                for old in prior:
                    if old['order'] == latest_order:
                        f['previous'].append(old['id'])
                        old['next'].append(f['id'])
            chain.append(f)
        out.append(f)
    return out


def build_memory(sessions, workers=8):
    api = API()
    extracted = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(extract_session, s, api): s['id'] for s in sessions}
        for job in as_completed(jobs):
            sid = jobs[job]
            extracted[sid] = job.result()
            print('extracted', sid, len(extracted[sid]['facts']), flush=True)
    facts = [f for s in sessions for f in extracted[s['id']]['facts']]
    return dict(sessions=sessions, facts=evolve(facts), extraction=extracted,
                schema='vem-0.1', extraction_prompt_hash=digest(EXTRACT))


class Memory:
    def __init__(self, data):
        self.data = data
        self.turns = [t for s in data['sessions'] for t in s['turns']]
        self.by_turn = {t['id']: t for t in self.turns}
        self.turn_pos = {t['id']: i for i, t in enumerate(self.turns)}
        self.facts = data.get('facts', [])
        self.by_fact = {f['id']: f for f in self.facts}
        self.raw_index = BM25([t['speaker'] + ' ' + t['text'] for t in self.turns])
        self.fact_index = BM25([f.get('subject','') + ' ' + f.get('slot','') + ' ' + f['text'] for f in self.facts])
        self.tracks = [t for t in data.get('tracks', [])
                       if isinstance(t.get('entity'), str) and isinstance(t.get('attribute'), str)]
        self.track_index = BM25([t['entity'] + ' ' + t['attribute'] + ' ' +
                                 ' '.join(self.by_fact[mid]['text'] for mid in t['members'])
                                 for t in self.tracks])

    def retrieve(self, question, method, budget, encoder, dense=None):
        if method == 'hybrid_packed':
            return self.retrieve_packed(question, budget, encoder, dense)
        if method == 'full':
            parts = [self.render_turn(t) for t in self.turns]
            text = '\n'.join(parts)
            return dict(text=text, source_ids=list(self.by_turn), facts=[], tokens=len(encoder.encode(text)))
        raw_rank = self.raw_index.rank(question)
        fact_rank = self.fact_index.rank(question)
        if method in ('hybrid', 'hybrid_facts', 'hybrid_notes', 'consolidated_hybrid'):
            def fuse(lexical, semantic):
                scores = defaultdict(float)
                for ranking in (lexical, semantic):
                    for rank, (i, score) in enumerate(ranking):
                        scores[i] += 1 / (40 + rank)
                return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            raw_rank = fuse(raw_rank, dense['raw'])
            fact_rank = fuse(fact_rank, dense['facts'])
        candidates = []
        if method in ('raw', 'hybrid'):
            candidates = [('turn', self.turns[i]['id']) for i, score in raw_rank]
        else:
            # Reciprocal-rank fusion of direct evidence and semantic facts.
            scores = defaultdict(float)
            for rank, (i, score) in enumerate(raw_rank):
                scores[('turn', self.turns[i]['id'])] += 1 / (40 + rank)
            for rank, (i, score) in enumerate(fact_rank):
                scores[('fact', self.facts[i]['id'])] += 1 / (40 + rank)
            candidates = sorted(scores, key=lambda key: -scores[key])
        parts, source_ids, fact_ids, seen = [], set(), [], set()
        used = 0

        def add(kind, mid, render_evolution=False):
            nonlocal used
            if (kind, mid) in seen:
                return
            seen.add((kind, mid))
            if kind == 'turn':
                text = self.render_turn(self.by_turn[mid])
                refs = [mid]
            else:
                f = self.by_fact[mid]
                text = '[{}; observed {}; event {}] {}'.format(mid, f['observed_at'], f.get('event_date') or 'unspecified', f['text'])
                if render_evolution and f.get('next'):
                    text += ' [Later state observations: {}]'.format(', '.join(f['next']))
                refs = f['sources']
                # Include direct source alongside compressed fact, preserving attribution.
                if method == 'hybrid_notes':
                    text += ' [sources: {}]'.format(', '.join(refs))
                else:
                    text += '\n' + '\n'.join(self.render_turn(self.by_turn[r]) for r in refs if r not in source_ids)
            size = len(encoder.encode(text + '\n'))
            if used + size > budget:
                return
            used += size
            parts.append(text)
            source_ids.update(refs)
            if kind == 'fact':
                fact_ids.append(mid)

        if method in ('consolidated', 'consolidated_hybrid') and re.search(r'\b(current|currently|latest|now)\b|most recent', question.lower()):
            # Updated track membership exposes newer witnesses even if their wording differs.
            for ti, score in self.track_index.rank(question, 2):
                for mid in reversed(self.tracks[ti]['members'][-6:]):
                    add('fact', mid)
        for rank, (kind, mid) in enumerate(candidates):
            add(kind, mid, method == 'evolved')
            if method == 'evolved' and rank < 16:
                if kind == 'fact':
                    f = self.by_fact[mid]
                    # State-chain closure includes both historical and newer observations.
                    for link in f.get('next', []) + f.get('previous', []):
                        add('fact', link, True)
                else:
                    # Nearby turns resolve references without importing an entire session.
                    pos = self.turn_pos[mid]
                    for p in (pos - 1, pos + 1):
                        if 0 <= p < len(self.turns) and self.turns[p]['session'] == self.by_turn[mid]['session']:
                            add('turn', self.turns[p]['id'])
        text = '\n'.join(parts)
        return dict(text=text, source_ids=sorted(source_ids), facts=fact_ids, tokens=len(encoder.encode(text)))

    def retrieve_packed(self, question, budget, encoder, dense):
        # Rank every view at its immutable source, rather than spending context on duplicates.
        scores = defaultdict(float)
        rankings = [(self.raw_index.rank(question), False), (dense['raw'], False),
                    (self.fact_index.rank(question), True), (dense['facts'], True)]
        for ranking, is_fact in rankings:
            seen = set()
            for rank, (i, score) in enumerate(ranking):
                refs = self.facts[i]['sources'] if is_fact else [self.turns[i]['id']]
                for reference in refs:
                    if reference not in seen:
                        scores[reference] += 1 / (40 + rank)
                        seen.add(reference)
        selected, sessions = [], set()
        used = 0
        for reference in sorted(scores, key=lambda mid: (-scores[mid], self.turn_pos[mid])):
            t = self.by_turn[reference]
            line = '[{}] {}: {}'.format(reference, t['speaker'], t['text'])
            header = ('Session {} | {}\n'.format(t['session'], t['date'])
                      if t['session'] not in sessions else '')
            cost = len(encoder.encode(header + line + '\n'))
            if used + cost <= budget:
                selected.append(reference)
                sessions.add(t['session'])
                used += cost
        parts, current_session = [], None
        for reference in sorted(selected, key=self.turn_pos.get):
            t = self.by_turn[reference]
            if current_session != t['session']:
                current_session = t['session']
                parts.append('Session {} | {}'.format(t['session'], t['date']))
            parts.append('[{}] {}: {}'.format(reference, t['speaker'], t['text']))
        text = '\n'.join(parts)
        return dict(text=text, source_ids=selected, facts=[], tokens=len(encoder.encode(text)))

    @staticmethod
    def render_turn(t):
        return '[{} | {} | {}] {}'.format(t['id'], t['date'], t['speaker'], t['text'])
