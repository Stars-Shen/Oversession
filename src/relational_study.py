"""One fixed 80-question relation-memory study; no reader or model sweep."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import threading
from types import SimpleNamespace
import numpy as np
import tiktoken
from api import ROOT, atomic_json, digest
from full_history_reader import CONTROL, read_rows
from memory import Memory
from openai_flex import FlexAPI
from relational_memory import PROMPT, RelationMemory, active_cards, build_ledger

FOLDER = ROOT / 'runs/phase3/relational_dev'
REFERENCE = ROOT / 'runs/phase3/openai_flex_reasoning_dev'


class StudyAPI(FlexAPI):
    """Reserve a conservative per-request estimate before concurrent calls."""
    def __init__(self, spent):
        super().__init__(); self.reserved = spent; self.lock = threading.Lock()

    def call(self, instructions, content, max_tokens=4096, **kwargs):
        # UTF-8 bytes + overhead is a planning proxy, not a tokenizer guarantee.
        quote = ((len((instructions + content).encode()) + 1024) * .0625 + max_tokens * .25) / 1e6
        with self.lock:
            assert self.reserved + quote <= .50, 'Stop at the prospective $0.50 planning limit'
            self.reserved += quote
        result = super().call(instructions, content, max_tokens=max_tokens, **kwargs)
        with self.lock:
            self.reserved += result['estimated_new_cost_usd'] - quote
        if result['new_paid_call'] and result.get('served_service_tier') != 'flex':
            raise RuntimeError('Unconfirmed Flex service; stop subsequent batches')
        return result


def artifacts():
    config = json.loads((FOLDER / 'config.json').read_text())
    memories = {}
    for name, expected in config['memory_sha256'].items():
        path = ROOT / 'runs/memories_captions' / (name + '.json')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        memories[name] = json.loads(path.read_text())
    assert config['writer_prompt_hash'] == digest(PROMPT)
    for filename, expected in config['implementation_sha256'].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected
    return config, memories


def records():
    path = FOLDER / 'predictions.jsonl'
    return read_rows(path) if path.exists() else []


def costs():
    steps = [s for p in (FOLDER / 'ledgers').glob('*.json')
             for s in json.loads(p.read_text())['steps']]
    all_calls = steps + records()
    return dict(estimated_new_cost_usd=sum(s['estimated_new_cost_usd'] for s in all_calls),
                new_paid_calls=sum(s['new_paid_call'] for s in all_calls))


def prepare():
    old = json.loads((CONTROL / 'config.json').read_text()); rows = read_rows(CONTROL / 'inputs.jsonl')
    low = json.loads((REFERENCE / 'baseline_low.json').read_text())
    assert len(rows) == 80 and {r['qid'] for r in rows} == {r['qid'] for r in low}
    config = dict(question_ids=[r['qid'] for r in rows], original_inputs_hash=digest(rows),
                  memory_sha256=old['memory_sha256'], writer_prompt_hash=digest(PROMPT),
                  model='gpt-6-luna', service_tier='flex', reasoning='low', max_new_tokens=768,
                  context_budget=3000, planning_limit_usd=.50, policy='relational-v1',
                  split='reused fixed 80 development questions; no held-out claim',
                  frozen='extraction, reader system, reasoning, output cap, raw/fact dense embeddings',
                  decision='>=3 F1 points above strongest cached reader, <=5% incomplete, before separate 160',
                  dataset_sha256=hashlib.sha256((ROOT / 'data/locomo10.json').read_bytes()).hexdigest(),
                  scorer_sha256=hashlib.sha256((ROOT / 'third_party/locomo_evaluation.py').read_bytes()).hexdigest(),
                  implementation_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                      for p in ('src/relational_memory.py', 'src/relational_study.py')})
    path = FOLDER / 'config.json'
    if path.exists():
        assert json.loads(path.read_text()) == config
    atomic_json(path, config)
    atomic_json(FOLDER / 'baseline_low.json', low)
    print('prepared fixed 80; no model calls', flush=True)


def build():
    _, memories = artifacts(); api = StudyAPI(costs()['estimated_new_cost_usd'])
    # Sessions within each memory are strictly sequential; no future facts supplied.
    def one(name):
        memory = memories[name]
        build_ledger(memory, api, FOLDER / 'ledgers' / (name + '.json'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(one, memories))
    print(costs(), flush=True)


def contexts():
    config, data = artifacts(); original = read_rows(CONTROL / 'inputs.jsonl')
    assert digest(original) == config['original_inputs_hash']
    os.environ['TIKTOKEN_CACHE_DIR'] = str(ROOT / 'runs/tokenizer_cache')
    encoder = SimpleNamespace(encode=tiktoken.get_encoding('cl100k_base').encode_ordinary)
    memories = {name: Memory(d) for name, d in data.items()}
    ledgers = {name: json.loads((FOLDER / 'ledgers' / (name + '.json')).read_text()) for name in data}
    assert all(len(ledgers[n]['steps']) == len(data[n]['sessions']) for n in data)
    indices = {n: RelationMemory(memories[n], ledgers[n]['cards']) for n in data}
    embeddings = {n: np.load(ROOT / 'runs/embeddings_captions' / (n + '.npz')) for n in data}
    for name, emb in embeddings.items():
        assert list(emb['document_ids']) == [t['id'] for t in memories[name].turns]
        assert list(emb['fact_ids']) == [f['id'] for f in memories[name].facts]
    rows = []
    for row in original:
        name, qi = row['qid'].rsplit(':', 1); emb = embeddings[name]
        vector = emb['queries'][int(qi)].astype(np.float32)
        dense = {k: sorted(enumerate(emb[k].astype(np.float32) @ vector), key=lambda x: (-x[1], x[0]))
                 for k in ('raw', 'facts')}
        old = memories[name].retrieve_packed(row['question'], 3000, encoder, dense)
        assert old['text'] == row['context'] and old['source_ids'] == row['source_ids']
        result = indices[name].retrieve(row['question'], 3000, encoder, dense)
        assert result['tokens'] <= 3000
        assert all(set(r['sources']) <= set(result['source_ids']) for r in result.get('relations', []))
        rows.append(dict(row, context=result['text'], context_tokens=result['tokens'],
                         source_ids=result['source_ids'], relations=result.get('relations', [])))
    (FOLDER / 'inputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    config.update(inputs_hash=digest(rows), ledger_hash={n: digest(ledgers[n]) for n in ledgers})
    atomic_json(FOLDER / 'config.json', config)
    print(dict(contexts=len(rows), eligible=sum(bool(r['relations']) for r in rows),
               baseline_reproduction='80/80 exact', active_cards={n: len(active_cards(l['cards'])) for n, l in ledgers.items()}), flush=True)


def generate():
    config, _ = artifacts(); rows = read_rows(FOLDER / 'inputs.jsonl')
    assert digest(rows) == config['inputs_hash']
    for name, expected in config['ledger_hash'].items():
        assert digest(json.loads((FOLDER / 'ledgers' / (name + '.json')).read_text())) == expected
    saved = records(); done = {r['qid'] for r in saved}
    assert len(saved) == len(done) and done <= set(config['question_ids'])
    pending = [r for r in rows if r['qid'] not in done]
    api = StudyAPI(costs()['estimated_new_cost_usd'])
    def answer(row):
        result = api.call(row['system'], 'Memories:\n' + row['context'] + '\n\nQuestion: ' + row['question'],
                          max_tokens=768, effort='low')
        return dict(qid=row['qid'], method='relations', hypothesis=result['text'], complete=result.get('complete', True),
                    source_ids=row['source_ids'], context_tokens=row['context_tokens'], relations=row['relations'],
                    usage=result['usage'], request_hash=result['request_hash'], new_paid_call=result['new_paid_call'],
                    estimated_new_cost_usd=result['estimated_new_cost_usd'], served_service_tier=result.get('served_service_tier'))
    with ThreadPoolExecutor(max_workers=4) as pool, (FOLDER / 'predictions.jsonl').open('a') as out:
        for start in range(0, len(pending), 4):
            for result in pool.map(answer, pending[start:start + 4]):
                out.write(json.dumps(result, ensure_ascii=False) + '\n'); out.flush(); saved.append(result)
                print(len(saved), '/80', result['qid'], repr(result['hypothesis']), flush=True)
    print(costs(), flush=True)


def score():
    from experiment import scorer, summarize
    config, _ = artifacts(); candidate = records(); ids = config['question_ids']
    assert len(candidate) == 80 and {r['qid'] for r in candidate} == set(ids)
    for path, expected in [('data/locomo10.json', 'dataset_sha256'),
                           ('third_party/locomo_evaluation.py', 'scorer_sha256')]:
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == config[expected]
    data = {s['sample_id']: s for s in json.loads((ROOT / 'data/locomo10.json').read_text())}
    low = json.loads((FOLDER / 'baseline_low.json').read_text())
    medium = read_rows(REFERENCE / 'predictions.jsonl')
    scored = []; metric = scorer()
    for method, rows in [('relations', candidate), ('luna_low', low), ('luna_medium', medium)]:
        assert len(rows) == 80 and {r['qid'] for r in rows} == set(ids)
        for row in rows:
            name, qi = row['qid'].rsplit(':', 1); q = data[name]['qa'][int(qi)]
            gold = {mid for e in q.get('evidence', []) for mid in re.findall(r'D\d+:\d+', str(e))}
            scored.append(dict(row, method=method, category=q['category'], f1=metric(row['hypothesis'], q),
                               evidence_recall=len(gold & set(row['source_ids'])) / len(gold) if gold else None))
    metrics = summarize(scored); comparisons = {}
    for control in ('luna_low', 'luna_medium'):
        a = {r['qid']: r['f1'] for r in scored if r['method'] == 'relations'}
        b = {r['qid']: r['f1'] for r in scored if r['method'] == control}
        delta = np.array([a[qid] - b[qid] for qid in ids])
        ci = np.quantile(np.random.default_rng(20261111).choice(delta, (5000, 80), replace=True).mean(1), [.025, .975])
        comparisons[control] = dict(delta=float(delta.mean()), bootstrap_ci=ci.tolist(),
                                   wins=int((delta > 0).sum()), losses=int((delta < 0).sum()), ties=int((delta == 0).sum()))
    incomplete = [r['qid'] for r in candidate if not r['complete']]
    report = dict(complete=True, n=80, config=config, metrics=metrics, comparisons=comparisons,
                  incomplete_outputs=incomplete, cost=costs(),
                  contexts_with_relations=sum(bool(r['relations']) for r in candidate),
                  expansion_gate_passed=comparisons['luna_medium']['delta'] >= .03 and len(incomplete) <= 4,
                  interpretation='Fixed extraction and Luna low reader; reused development; relation provenance is structural, not a semantic proof. No SOTA claim.')
    atomic_json(FOLDER / 'scored.json', scored); atomic_json(ROOT / 'reports/relational_dev.json', report)
    print(json.dumps(dict(f1={m: r['f1'] for m, r in metrics.items()}, comparisons=comparisons,
                          cost=costs(), gate=report['expansion_gate_passed']), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'build', 'contexts', 'generate', 'score'])
    args = parser.parse_args(); FOLDER.mkdir(parents=True, exist_ok=True); globals()[args.command]()
