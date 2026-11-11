"""One global-index selection call, then the unchanged Luna answerer."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import os
import re
from types import SimpleNamespace
import numpy as np
from api import ROOT, atomic_json, digest
from full_history_reader import CONTROL, read_rows
from joint_selection import PROMPT, memory_index, selected_context
from memory import Memory
from openai_flex import FlexAPI

FOLDER = ROOT / 'runs/phase3/joint_selection_dev'
REFERENCE = ROOT / 'runs/phase3/openai_flex_reasoning_dev'


def records(stage):
    path = FOLDER / (stage + '.jsonl')
    return read_rows(path) if path.exists() else []


def costs():
    return {stage: dict(new_paid_calls=sum(r['new_paid_call'] for r in records(stage)),
                       estimated_new_cost_usd=sum(r['estimated_new_cost_usd'] for r in records(stage)))
            for stage in ('selections', 'predictions')}


def checked_config():
    config = json.loads((FOLDER / 'config.json').read_text())
    for group in ('memory_sha256', 'code_sha256'):
        for path, sha in config[group].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha
    assert digest(read_rows(CONTROL / 'inputs.jsonl')) == config['original_inputs_hash']
    return config


def selector_content(row, indexes):
    name = row['qid'].rsplit(':', 1)[0]
    lines = ['Fact rows: alias | observed date | event date (? = unknown) | subject | slot | text',
             'Residual rows: alias | observed date | speaker | original text']
    for entry in indexes[name]['entries']:
        keys = ('id', 'observed_at', 'event_date', 'subject', 'slot', 'text') if entry['id'].startswith('F') else ('id', 'date', 'speaker', 'text')
        lines.append(' | '.join(str(entry[k]) if entry[k] is not None else '?' for k in keys))
    return 'Memory index:\n' + '\n'.join(lines) + '\nQuestion: ' + row['question']


def prepare():
    original = json.loads((CONTROL / 'config.json').read_text()); rows = read_rows(CONTROL / 'inputs.jsonl')
    assert len(rows) == 80 and digest(rows) == original['inputs_hash']
    indexes = {}; hashes = {}
    for name, expected in original['memory_sha256'].items():
        p = ROOT / 'runs/memories_captions' / (name + '.json')
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected
        hashes[str(p.relative_to(ROOT))] = expected
        indexes[name] = memory_index(Memory(json.loads(p.read_text())))
        atomic_json(FOLDER / 'indexes' / (name + '.json'), indexes[name])
    quote = sum(((len((PROMPT + selector_content(r, indexes)).encode()) + 1024) * .125 + 1536 * .5) / 1e6 for r in rows)
    assert quote < 1., 'Selector planning quote exceeds $1.00; do not send calls'
    config = dict(question_ids=original['question_ids'], original_inputs_hash=digest(rows),
        memory_sha256=hashes, index_hash={n: digest(i) for n, i in indexes.items()},
        code_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                     ('src/joint_selection.py', 'src/joint_selection_study.py', 'src/openai_flex.py')},
        dataset_sha256=hashlib.sha256((ROOT / 'data/locomo10.json').read_bytes()).hexdigest(),
        scorer_sha256=hashlib.sha256((ROOT / 'third_party/locomo_evaluation.py').read_bytes()).hexdigest(),
        model='gpt-6-luna', service_tier='default', selection_effort='low', selection_max_tokens=1536,
        reader_effort='low', reader_max_tokens=768, context_budget=3000, planning_limit_usd=1.,
        conservative_selection_quote_usd=quote, split='reused fixed 80 development questions',
        decision='Expand once only if >=3 F1 points above strongest cached reader and <=5% incomplete.',
        compute='One additional global-index lookup; final reader <=3K, total inference tokens are higher.')
    path = FOLDER / 'config.json'
    if path.exists():
        assert json.loads(path.read_text()) == config
    atomic_json(path, config)
    print(dict(prepared=80, conservative_selection_quote_usd=quote, index_entries={n: len(i['entries']) for n, i in indexes.items()}), flush=True)


def run(stage):
    config = checked_config(); api = FlexAPI(service_tier='default')
    indexes = {n: json.loads((FOLDER / 'indexes' / (n + '.json')).read_text()) for n in config['index_hash']}
    assert all(digest(indexes[n]) == h for n, h in config['index_hash'].items())
    rows = read_rows(CONTROL / 'inputs.jsonl') if stage == 'selections' else read_rows(FOLDER / 'inputs.jsonl')
    if stage == 'predictions':
        assert digest(rows) == config['inputs_hash']
    saved = records(stage); done = {r['qid'] for r in saved}
    assert len(done) == len(saved) and done <= set(config['question_ids'])
    pending = [r for r in rows if r['qid'] not in done]
    def request(row):
        if stage == 'selections':
            return PROMPT, selector_content(row, indexes), 1536, True
        return row['system'], 'Memories:\n' + row['context'] + '\n\nQuestion: ' + row['question'], 768, False
    quote = sum(((len((s + content).encode()) + 1024) * .125 + cap * .5) / 1e6
                for s, content, cap, structured in (request(r) for r in pending))
    assert sum(s['estimated_new_cost_usd'] for s in costs().values()) + quote <= 1.
    def one(row):
        system, content, cap, structured = request(row)
        result = api.call(system, content, max_tokens=cap, effort='low', json_output=structured)
        if result['new_paid_call'] and result.get('served_service_tier') != 'default':
            raise RuntimeError('Unconfirmed Standard tier; stop following batches')
        out = dict(qid=row['qid'], complete=result.get('complete', True), usage=result['usage'],
                   request_hash=result['request_hash'], served_service_tier=result.get('served_service_tier'),
                   new_paid_call=result['new_paid_call'], estimated_new_cost_usd=result['estimated_new_cost_usd'])
        if structured:
            out['selection'] = result.get('parsed')
        else:
            out.update(method='joint_selection', hypothesis=result['text'], source_ids=row['source_ids'],
                       context_tokens=row['context_tokens'])
        return out
    with ThreadPoolExecutor(max_workers=4) as pool, (FOLDER / (stage + '.jsonl')).open('a') as out:
        for start in range(0, len(pending), 4):
            jobs = {pool.submit(one, r): r for r in pending[start:start + 4]}; errors = []
            for job in as_completed(jobs):
                try:
                    result = job.result()
                except Exception as error:
                    errors.append(dict(qid=jobs[job]['qid'], error=str(error))); continue
                out.write(json.dumps(result, ensure_ascii=False) + '\n'); out.flush(); saved.append(result)
            print(stage, len(saved), '/80', flush=True)
            if errors:
                atomic_json(FOLDER / (stage + '_failures.json'), errors)
                raise RuntimeError('Batch stopped; successful responses saved; no automatic retries')
    assert len(saved) == 80


def contexts():
    import tiktoken
    config = checked_config(); selections = {r['qid']: r for r in records('selections')}
    assert set(selections) == set(config['question_ids'])
    os.environ['TIKTOKEN_CACHE_DIR'] = str(ROOT / 'runs/tokenizer_cache')
    encoder = SimpleNamespace(encode=tiktoken.get_encoding('cl100k_base').encode_ordinary)
    memories = {Path(p).stem: Memory(json.loads((ROOT / p).read_text())) for p in config['memory_sha256']}
    indexes = {n: memory_index(m) for n, m in memories.items()}; result = []
    assert all(digest(indexes[n]) == h for n, h in config['index_hash'].items())
    for row in read_rows(CONTROL / 'inputs.jsonl'):
        name = row['qid'].rsplit(':', 1)[0]; selection = selections[row['qid']]
        new = selected_context(memories[name], indexes[name], selection['selection'], 3000, encoder) if selection['complete'] else None
        context = new or dict(text=row['context'], source_ids=row['source_ids'], tokens=row['context_tokens'])
        result.append(dict(row, context=context['text'], source_ids=context['source_ids'], context_tokens=context['tokens'],
                           selection_applied=new is not None, selected_aliases=context.get('selected_aliases', []), roots=context.get('roots', [])))
    config['inputs_hash'] = digest(result); atomic_json(FOLDER / 'config.json', config)
    (FOLDER / 'inputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in result))
    print(dict(contexts=80, applied=sum(r['selection_applied'] for r in result), mean_tokens=float(np.mean([r['context_tokens'] for r in result]))), flush=True)


def score():
    from experiment import scorer, summarize
    config = checked_config(); metric = scorer(); data_path = ROOT / 'data/locomo10.json'
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == config['dataset_sha256']
    assert hashlib.sha256((ROOT / 'third_party/locomo_evaluation.py').read_bytes()).hexdigest() == config['scorer_sha256']
    data = {s['sample_id']: s for s in json.loads(data_path.read_text())}
    groups = dict(joint_selection=records('predictions'), luna_low=json.loads((REFERENCE / 'baseline_low.json').read_text()), luna_medium=read_rows(REFERENCE / 'predictions.jsonl'))
    scored = []
    for method, rows in groups.items():
        assert len(rows) == 80 and {r['qid'] for r in rows} == set(config['question_ids'])
        for r in rows:
            name, qi = r['qid'].rsplit(':', 1); q = data[name]['qa'][int(qi)]
            gold = {mid for e in q.get('evidence', []) for mid in re.findall(r'D\d+:\d+', str(e))}
            scored.append(dict(r, method=method, category=q['category'], f1=metric(r['hypothesis'], q),
                               evidence_recall=len(gold & set(r['source_ids'])) / len(gold) if gold else None))
    comparisons = {}; candidate = {r['qid']: r['f1'] for r in scored if r['method'] == 'joint_selection'}
    for control in ('luna_low', 'luna_medium'):
        reference = {r['qid']: r['f1'] for r in scored if r['method'] == control}
        delta = np.array([candidate[q] - reference[q] for q in config['question_ids']])
        ci = np.quantile(np.random.default_rng(20261111).choice(delta, (5000, 80), replace=True).mean(1), [.025, .975])
        comparisons[control] = dict(delta=float(delta.mean()), bootstrap_ci=ci.tolist(), wins=int((delta > 0).sum()), losses=int((delta < 0).sum()), ties=int((delta == 0).sum()))
    incomplete = [r['qid'] for r in groups['joint_selection'] if not r['complete']]
    report = dict(complete=True, n=80, config=config, metrics=summarize(scored), comparisons=comparisons,
        cost=costs(), total_estimated_new_cost_usd=sum(s['estimated_new_cost_usd'] for s in costs().values()),
        incomplete_outputs=incomplete, selector_incomplete=[r['qid'] for r in records('selections') if not r['complete']],
        applied=sum(r['selection_applied'] for r in read_rows(FOLDER / 'inputs.jsonl')),
        expansion_gate_passed=comparisons['luna_medium']['delta'] >= .03 and len(incomplete) <= 4,
        interpretation='One extra global-index lookup; fixed final reader and max evidence budget; extra inference compute; reused development, not SOTA.')
    atomic_json(FOLDER / 'scored.json', scored); atomic_json(ROOT / 'reports/joint_selection_dev.json', report)
    print(json.dumps(dict(f1={m: r['f1'] for m, r in report['metrics'].items()}, comparisons=comparisons,
                         cost=report['total_estimated_new_cost_usd'], gate=report['expansion_gate_passed']), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'select', 'contexts', 'generate', 'score'])
    args = parser.parse_args(); FOLDER.mkdir(parents=True, exist_ok=True)
    if args.command in ('select', 'generate'):
        run('selections' if args.command == 'select' else 'predictions')
    else:
        globals()[args.command]()
