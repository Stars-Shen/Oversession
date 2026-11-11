"""Fixed turn-writer comparison, preserving baseline data and reader semantics."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import numpy as np
from api import ROOT, atomic_json, digest
from full_history_reader import CONTROL, read_rows
from memory import Memory
from openai_flex import FlexAPI
from turn_writer import PROMPT, inputs, observations, assemble

FOLDER = ROOT / 'runs/phase3/turn_dev'
REFERENCE = ROOT / 'runs/phase3/openai_flex_reasoning_dev'


def rows(filename):
    path = FOLDER / filename
    return read_rows(path) if path.exists() else []


def costs():
    records = rows('writer_records.jsonl') + rows('predictions.jsonl')
    return dict(estimated_new_cost_usd=sum(r['estimated_new_cost_usd'] for r in records),
                new_paid_calls=sum(r['new_paid_call'] for r in records))


class BudgetAPI(FlexAPI):
    def __init__(self):
        super().__init__(service_tier='default'); self.reserved = costs()['estimated_new_cost_usd']; self.lock = threading.Lock()

    def call(self, instructions, content, max_tokens=2048, **kwargs):
        quote = ((len((instructions + content).encode()) + 1024) * .125 + max_tokens * .50) / 1e6
        with self.lock:
            assert self.reserved + quote <= 2.50, 'Stop at $2.50 planning limit'
            self.reserved += quote
        result = super().call(instructions, content, max_tokens=max_tokens, **kwargs)
        with self.lock:
            self.reserved += result['estimated_new_cost_usd'] - quote
        if result['new_paid_call'] and result.get('served_service_tier') != 'default':
            raise RuntimeError('Unconfirmed Standard service; stop further batches')
        return result


def artifacts():
    config = json.loads((FOLDER / 'config.json').read_text())
    for group in ('memory_sha256', 'embedding_sha256', 'implementation_sha256'):
        for path, expected in config[group].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    return config


def prepare():
    original = json.loads((CONTROL / 'config.json').read_text()); names = sorted(original['memory_sha256'])
    writer = []; paths = [p for name in names for p in
        ('runs/memories_captions/' + name + '.json', 'runs/embeddings_captions/' + name + '.npz')]
    for name in names:
        data = json.loads((ROOT / 'runs/memories_captions' / (name + '.json')).read_text())
        writer.extend(dict(row, name=name) for row in inputs(data))
    config = dict(model='gpt-6-luna', service_tier='default', writer_effort='low', writer_max_tokens=2048,
                  reader_effort='low', reader_max_tokens=768, context_budget=3000,
                  question_ids=original['question_ids'], original_inputs_hash=original['inputs_hash'],
                  writer_inputs_hash=digest(writer), writer_targets=len(writer), planning_limit_usd=2.50,
                  writer_prompt_hash=digest(PROMPT), split='reused fixed 80 development',
                  memory_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths if p.endswith('.json')},
                  embedding_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths if p.endswith('.npz')},
                  implementation_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                      for p in ('src/turn_writer.py', 'src/turn_study.py', 'src/openai_flex.py')},
                  dataset_sha256=hashlib.sha256((ROOT / 'data/locomo10.json').read_bytes()).hexdigest(),
                  scorer_sha256=hashlib.sha256((ROOT / 'third_party/locomo_evaluation.py').read_bytes()).hexdigest(),
                  policy='Target + preceding 8 turns + up to 8 frozen earlier-session facts; no newly written belief feedback',
                  decision='>=3 F1 points above strongest cached reader and <=5% incomplete before separate 160')
    quote = sum(((len((PROMPT + json.dumps(r['payload'], ensure_ascii=False)).encode()) + 1024) * .125 + 2048 * .50) / 1e6 for r in writer)
    assert quote + .20 <= 2.50
    config['conservative_writer_quote_usd'] = quote
    path = FOLDER / 'config.json'
    if path.exists():
        assert json.loads(path.read_text()) == config
    atomic_json(path, config)
    (FOLDER / 'writer_inputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in writer))
    print(dict(targets=len(writer), conservative_writer_quote_usd=quote, paid_calls=0), flush=True)


def build():
    config = artifacts(); writer = rows('writer_inputs.jsonl')
    assert digest(writer) == config['writer_inputs_hash']
    saved = rows('writer_records.jsonl'); done = {(r['name'], r['target_id']) for r in saved}
    assert len(done) == len(saved)
    pending = [r for r in writer if (r['name'], r['target_id']) not in done]; api = BudgetAPI()
    def extract(row):
        result = api.call(PROMPT, json.dumps(row['payload'], ensure_ascii=False),
                          max_tokens=2048, json_output=True, effort='low')
        complete = result.get('complete', True)
        facts, rejected = observations(row, result['parsed']['facts']) if complete else ([], [])
        return dict(name=row['name'], target_id=row['target_id'], order=row['order'], facts=facts,
                    rejected=rejected, complete=complete, usage=result['usage'], request_hash=result['request_hash'],
                    served_service_tier=result.get('served_service_tier'),
                    estimated_new_cost_usd=result['estimated_new_cost_usd'], new_paid_call=result['new_paid_call'])
    with ThreadPoolExecutor(max_workers=8) as pool, (FOLDER / 'writer_records.jsonl').open('a') as out:
        for start in range(0, len(pending), 8):
            jobs = {pool.submit(extract, r): r for r in pending[start:start + 8]}; errors = []
            for job in as_completed(jobs):
                try:
                    result = job.result()
                except Exception as error:
                    r = jobs[job]; errors.append(dict(name=r['name'], target_id=r['target_id'], error=str(error)))
                    continue
                out.write(json.dumps(result, ensure_ascii=False) + '\n'); out.flush(); saved.append(result)
            print('writer', len(saved), '/', len(writer), 'facts', sum(len(r['facts']) for r in saved), flush=True)
            if errors:
                atomic_json(FOLDER / 'writer_failures.json', errors)
                raise RuntimeError('Writer batch failed; successful responses saved; no automatic retries')
            assert sum(not r['complete'] for r in saved) <= .05 * len(writer), 'Stop: over 5% incomplete'
    assert len(saved) == len(writer) and {(r['name'], r['target_id']) for r in saved} == {(r['name'], r['target_id']) for r in writer}
    for p in config['memory_sha256']:
        data = json.loads((ROOT / p).read_text()); name = Path(p).stem
        atomic_json(FOLDER / 'memories' / (name + '.json'), assemble(data, [r for r in saved if r['name'] == name]))
    print(costs(), flush=True)


def encode():
    import torch
    from sentence_transformers import SentenceTransformer
    config = artifacts()
    model = SentenceTransformer('/home/wqshen/models/Qwen3-Embedding-0.6B', device='cuda:0',
                                model_kwargs={'torch_dtype': torch.float16}, local_files_only=True)
    model.max_seq_length = 2048
    for p in config['memory_sha256']:
        name = Path(p).stem; memory = json.loads((FOLDER / 'memories' / (name + '.json')).read_text())
        old = np.load(ROOT / 'runs/embeddings_captions' / (name + '.npz'))
        facts = model.encode([f['text'] for f in memory['facts']], batch_size=32,
                             normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
        target = FOLDER / 'embeddings' / (name + '.npz'); target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, raw=old['raw'], queries=old['queries'], facts=facts.astype(np.float16),
                            document_ids=old['document_ids'], fact_ids=np.array([f['id'] for f in memory['facts']]),
                            memory_hash=digest(memory))
        print('embedded new facts only', name, len(facts), flush=True)
    atomic_json(FOLDER / 'embedding_runtime.json', dict(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, model='Qwen3-Embedding-0.6B', dtype='float16'))


def contexts():
    import tiktoken
    config = artifacts(); original = read_rows(CONTROL / 'inputs.jsonl')
    assert digest(original) == config['original_inputs_hash']
    os.environ['TIKTOKEN_CACHE_DIR'] = str(ROOT / 'runs/tokenizer_cache')
    encoder = SimpleNamespace(encode=tiktoken.get_encoding('cl100k_base').encode_ordinary)
    memories = {}; old_memories = {}; embeddings = {}; old_embeddings = {}
    for p in config['memory_sha256']:
        name = Path(p).stem; old_data = json.loads((ROOT / p).read_text())
        data = json.loads((FOLDER / 'memories' / (name + '.json')).read_text())
        assert data['sessions'] == old_data['sessions']
        memories[name] = Memory(data); old_memories[name] = Memory(old_data)
        embeddings[name] = np.load(FOLDER / 'embeddings' / (name + '.npz'))
        old_embeddings[name] = np.load(ROOT / 'runs/embeddings_captions' / (name + '.npz'))
        assert str(embeddings[name]['memory_hash']) == digest(data)
        assert list(embeddings[name]['fact_ids']) == [f['id'] for f in data['facts']]
        for k in ('raw', 'queries', 'document_ids'):
            assert np.array_equal(embeddings[name][k], old_embeddings[name][k])
    result = []
    for row in original:
        name, qi = row['qid'].rsplit(':', 1)
        def context(memory, emb):
            vector = emb['queries'][int(qi)].astype(np.float32)
            dense = {k: sorted(enumerate(emb[k].astype(np.float32) @ vector), key=lambda x: (-x[1], x[0])) for k in ('raw', 'facts')}
            return memory.retrieve_packed(row['question'], 3000, encoder, dense)
        old = context(old_memories[name], old_embeddings[name])
        assert old['text'] == row['context'] and old['source_ids'] == row['source_ids']
        new = context(memories[name], embeddings[name]); assert new['tokens'] <= 3000
        result.append(dict(row, context=new['text'], source_ids=new['source_ids'], context_tokens=new['tokens']))
    (FOLDER / 'inputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in result))
    config['inputs_hash'] = digest(result); atomic_json(FOLDER / 'config.json', config)
    print('80 exact original controls; 80 new contexts prepared', flush=True)


def generate():
    config = artifacts(); reader = rows('inputs.jsonl'); assert digest(reader) == config['inputs_hash']
    saved = rows('predictions.jsonl'); done = {r['qid'] for r in saved}
    assert len(saved) == len(done) and done <= set(config['question_ids'])
    api = BudgetAPI()
    def answer(row):
        result = api.call(row['system'], 'Memories:\n' + row['context'] + '\n\nQuestion: ' + row['question'], max_tokens=768, effort='low')
        return dict(qid=row['qid'], method='turn_writer', hypothesis=result['text'], complete=result.get('complete', True),
                    source_ids=row['source_ids'], context_tokens=row['context_tokens'], usage=result['usage'],
                    request_hash=result['request_hash'], served_service_tier=result.get('served_service_tier'),
                    estimated_new_cost_usd=result['estimated_new_cost_usd'], new_paid_call=result['new_paid_call'])
    pending = [r for r in reader if r['qid'] not in done]
    with ThreadPoolExecutor(max_workers=4) as pool, (FOLDER / 'predictions.jsonl').open('a') as out:
        for start in range(0, len(pending), 4):
            for r in pool.map(answer, pending[start:start + 4]):
                out.write(json.dumps(r, ensure_ascii=False) + '\n'); out.flush(); saved.append(r)
                print(len(saved), '/80', r['qid'], repr(r['hypothesis']), flush=True)


def score():
    from experiment import scorer, summarize
    from full_history_reader import score as compare
    compare(FOLDER, 'turn_writer', 'New causal turn-anchored fact view; frozen four-view source RRF, Luna low/768 and 3K raw evidence. Reused development; not full PreCoMem.')
    report_path = ROOT / 'reports/turn_dev.json'; report = json.loads(report_path.read_text())
    scored = json.loads((FOLDER / 'scored.json').read_text()); data = {s['sample_id']: s for s in json.loads((ROOT / 'data/locomo10.json').read_text())}
    metric = scorer()
    for method, path in [('luna_low', REFERENCE / 'baseline_low.json'), ('luna_medium', REFERENCE / 'predictions.jsonl')]:
        control = json.loads(path.read_text()) if path.suffix == '.json' else read_rows(path)
        for r in control:
            name, qi = r['qid'].rsplit(':', 1); q = data[name]['qa'][int(qi)]
            import re
            gold = {mid for e in q.get('evidence', []) for mid in re.findall(r'D\d+:\d+', str(e))}
            scored.append(dict(r, method=method, category=q['category'], f1=metric(r['hypothesis'], q),
                               evidence_recall=len(gold & set(r['source_ids'])) / len(gold) if gold else None))
        a = {r['qid']: r['f1'] for r in scored if r['method'] == 'turn_writer'}; b = {r['qid']: r['f1'] for r in scored if r['method'] == method}
        assert set(a) == set(b)
        delta = np.array([a[qid] - b[qid] for qid in report['config']['question_ids']])
        ci = np.quantile(np.random.default_rng(20261111).choice(delta, (5000, 80), replace=True).mean(1), [.025, .975])
        report['comparisons'][method] = dict(delta=float(delta.mean()), bootstrap_ci=ci.tolist(), wins=int((delta > 0).sum()), losses=int((delta < 0).sum()), ties=int((delta == 0).sum()))
    report.update(metrics=summarize(scored), cost=costs(), new_paid_calls=costs()['new_paid_calls'],
                  expansion_gate_passed=report['comparisons']['luna_medium']['delta'] >= .03 and len(report['incomplete_outputs']) <= 4)
    atomic_json(FOLDER / 'scored.json', scored); atomic_json(report_path, report)
    print(dict(f1={m: s['f1'] for m, s in report['metrics'].items()}, cost=costs(), gate=report['expansion_gate_passed']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'build', 'encode', 'contexts', 'generate', 'score'])
    args = parser.parse_args(); FOLDER.mkdir(parents=True, exist_ok=True); globals()[args.command]()
