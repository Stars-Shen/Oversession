"""Question-blind indexing and paired LoCoMo experiments with pinned official F1."""
import argparse
import ast
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import re
import string
import numpy as np
import regex
from nltk.stem import PorterStemmer
import tiktoken
from api import API, ROOT, atomic_json, digest
from memory import Memory, locomo_sessions, build_memory
from state_reader import compile_state, PROMPT as STATE_PROMPT

ANSWER = '''Answer the question using the supplied dated conversation memories.
Give only the concise answer, without explanation, citations, preamble, or repetition of the question.
For a list, include all supported items separated by commas. Use the original names and precise details.
Resolve relative dates against the date of the source turn. Distinguish event time from the time it
was mentioned. Preserve historical facts; a later state matters only when the question asks about
that later state. Original turns are authoritative if a derived memory conflicts with them.
If the answer cannot be determined from these memories, say "No information available".
Do not guess missing personal facts.'''

COMPACT = '''
Output style: prefer the shortest sufficient noun phrase, usually 1-6 words, rather than a
complete sentence. Do not add biographical details or explanations that were not requested.
For a yes/no question, answer Yes, No, Likely yes, or Likely no as appropriate.
If the question explicitly asks what is likely or would happen, make a reasonable inference
from the person's stated preferences and experiences; direct verbatim evidence is not required.
Render exact dates as D Month YYYY, e.g. 28 January 2023, rather than ISO dates.
If only a month or year is supported, give just that precision.'''


def scorer():
    # Execute just the three unchanged official metric functions, avoiding BERTScore/torch imports.
    source = (ROOT / 'third_party/locomo_evaluation.py').read_text()
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('normalize_answer', 'f1_score', 'f1')]
    env = dict(regex=regex, string=string, ps=PorterStemmer(), Counter=Counter, np=np)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'official_locomo_f1', 'exec'), env)
    def score(prediction, question):
        cat = question['category']
        answer = str(question.get('answer', ''))
        if cat == 3:
            answer = answer.split(';')[0].strip()
        if cat == 5:
            return float(any(x in prediction.lower() for x in ('no information available', 'not mentioned')))
        return float(env['f1' if cat == 1 else 'f1_score'](prediction, answer))
    return score


def choose_questions(data, convs, limit, seed):
    groups = defaultdict(list)
    for ci in convs:
        sample = data[ci]
        for qi, q in enumerate(sample['qa']):
            if q['category'] != 5:
                groups[q['category']].append((ci, qi, q))
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    chosen = []
    while any(groups.values()) and (limit == 0 or len(chosen) < limit):
        for cat in sorted(groups):
            if groups[cat] and (limit == 0 or len(chosen) < limit):
                chosen.append(groups[cat].pop())
    return chosen


def summarize(records):
    result = {}
    for method in sorted({r['method'] for r in records}):
        rows = [r for r in records if r['method'] == method]
        cats = sorted({r['category'] for r in rows})
        result[method] = dict(n=len(rows), f1=float(np.mean([r['f1'] for r in rows])),
                             macro_f1=float(np.mean([np.mean([r['f1'] for r in rows if r['category'] == c]) for c in cats])),
                             evidence_recall=float(np.mean([r['evidence_recall'] for r in rows if r['evidence_recall'] is not None])),
                             context_tokens=float(np.mean([r['context_tokens'] for r in rows])),
                             by_category={c:dict(n=sum(r['category'] == c for r in rows),
                                             f1=float(np.mean([r['f1'] for r in rows if r['category'] == c]))) for c in cats})
    for baseline_method in ('raw', 'hybrid'):
        baseline = {r['qid']:r['f1'] for r in records if r['method'] == baseline_method}
        for method, summary in result.items():
            paired = [(r['f1'] - baseline[r['qid']]) for r in records if r['method'] == method and r['qid'] in baseline]
            if paired:
                d = np.array(paired)
                rng = np.random.default_rng(20261110)
                samples = rng.choice(d, (5000, len(d)), replace=True).mean(axis=1)
                summary['delta_vs_' + baseline_method] = dict(mean=float(d.mean()), ci95=np.quantile(samples,[.025,.975]).tolist(),
                                              wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum()))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['build', 'evaluate', 'summarize'])
    p.add_argument('--convs', default='0,1')
    p.add_argument('--limit', type=int, default=80)
    p.add_argument('--budget', type=int, default=3000)
    p.add_argument('--methods', default='raw,facts,evolved')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--run', default='pilot_v1')
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--reader', choices=['original', 'compact'], default='original')
    p.add_argument('--reasoning', choices=['low', 'medium'], default='low')
    p.add_argument('--captions', action='store_true')
    p.add_argument('--state-reader', action='store_true')
    args = p.parse_args()
    answer_prompt = ANSWER + (COMPACT if args.reader == 'compact' else '')
    memory_dir = ROOT / ('runs/memories_captions' if args.captions else 'runs/memories')
    embedding_dir = ROOT / ('runs/embeddings_captions' if args.captions else 'runs/embeddings')
    data_path = ROOT / 'data/locomo10.json'
    data = json.loads(data_path.read_text())
    convs = [int(x) for x in args.convs.split(',')]
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=True)
    if args.command == 'build':
        # A schema allowlist prevents QA, evidence labels, and gold answers entering the writer.
        sessions_by_conv = {ci:locomo_sessions({'conversation':data[ci]['conversation']}, args.captions) for ci in convs}
        for ci, sessions in sessions_by_conv.items():
            memory = build_memory(sessions, args.workers)
            memory['sample_id'] = data[ci]['sample_id']
            atomic_json(memory_dir / (data[ci]['sample_id'] + '.json'), memory)
            print('built', ci, 'facts', len(memory['facts']), 'version edges',sum(len(f['previous']) for f in memory['facts']), flush=True)
        return
    if args.command == 'summarize':
        rows = [json.loads(line) for line in (run / 'predictions.jsonl').read_text().splitlines()]
        summary = summarize(rows)
        atomic_json(run / 'summary.json', summary)
        print(json.dumps(summary, indent=2))
        return
    chosen = choose_questions(data, convs, args.limit, args.seed)
    methods = args.methods.split(',')
    memory_hashes = {}
    memories = {}
    embeddings = {}
    for ci in convs:
        mp = memory_dir / (data[ci]['sample_id'] + '.json')
        if methods == ['raw'] or methods == ['full']:
            memories[ci] = Memory(dict(sessions=locomo_sessions(data[ci], args.captions), facts=[]))
        else:
            memories[ci] = Memory(json.loads(mp.read_text()))
            memory_hashes[data[ci]['sample_id']] = hashlib.sha256(mp.read_bytes()).hexdigest()
        if any(m.startswith('hybrid') for m in methods):
            ep = embedding_dir / (data[ci]['sample_id'] + '.npz')
            embeddings[ci] = np.load(ep)
            if list(embeddings[ci]['document_ids']) != [t['id'] for t in memories[ci].turns]:
                raise ValueError('Embedding document IDs do not match memory')
            if list(embeddings[ci]['fact_ids']) != [f['id'] for f in memories[ci].facts]:
                raise ValueError('Embedding fact IDs do not match memory')
    config = dict(dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                  scorer_sha256=hashlib.sha256((ROOT / 'third_party/locomo_evaluation.py').read_bytes()).hexdigest(),
                  model='gpt-6-luna', answer_prompt_hash=digest(answer_prompt), convs=convs,
                  question_ids=[data[ci]['sample_id'] + ':' + str(qi) for ci,qi,q in chosen],
                  methods=methods, budget=args.budget, seed=args.seed, memory_sha256=memory_hashes,
                  split='development' if set(convs) <= {0,1} else ('full_mixed' if set(convs) & {0,1} else 'heldout'),
                  metric='official LoCoMo stemmed F1, categories 1-4; category 3 first answer before semicolon',
                  tokenizer='cl100k_base')
    if embeddings:
        config['embedding_model'] = 'Qwen3-Embedding-0.6B, float16 normalized, max_seq_length=2048'
        config['embedding_sha256'] = {data[ci]['sample_id']:hashlib.sha256((embedding_dir / (data[ci]['sample_id'] + '.npz')).read_bytes()).hexdigest() for ci in convs}
    if args.captions:
        config['caption_protocol'] = 'official blip_caption appended to dialogue; image search query excluded'
    if args.reasoning != 'low':
        config['reasoning_effort'] = args.reasoning
    if args.state_reader:
        config['state_reader_prompt_hash'] = digest(STATE_PROMPT)
    config_path = run / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Run config changed; choose a new --run name')
    atomic_json(config_path, config)
    os.environ['TIKTOKEN_CACHE_DIR'] = str(ROOT / 'runs/tokenizer_cache')
    encoder = tiktoken.get_encoding('cl100k_base')
    metric = scorer()
    records = []
    predfile = run / 'predictions.jsonl'
    if predfile.exists():
        records = [json.loads(line) for line in predfile.read_text().splitlines()]
    completed = {(r['qid'],r['method']) for r in records}
    api = API()

    def answer_one(ci, qi, q, method):
        dense = None
        if method.startswith('hybrid'):
            emb = embeddings[ci]
            vector = emb['queries'][qi].astype(np.float32)
            dense = {key:sorted(enumerate(emb[key].astype(np.float32) @ vector), key=lambda item: (-item[1], item[0])) for key in ('raw','facts')}
        context = memories[ci].retrieve(q['question'], method, args.budget, encoder, dense)
        state = None
        if args.state_reader:
            state = compile_state(api, q['question'], memories[ci].data['sessions'][-1]['date'],
                                  context, args.budget, encoder)
            context = memories[ci].retrieve(q['question'], method, args.budget-state['tokens']-32, encoder, dense)
            context['tokens'] += state['tokens']
        state_text = ('\nCompiled evidence state (advisory; original turns are authoritative):\n' + state['text']) if state else ''
        result = api.call(answer_prompt, 'Memories:\n' + context['text'] + state_text + '\n\nQuestion: ' + q['question'],
                          max_tokens=768 if args.reasoning == 'low' else 4096, effort=args.reasoning)
        gold = {mid for e in q.get('evidence', []) for mid in re.findall(r'D\d+:\d+', str(e))}
        recalled = set(context['source_ids'])
        return dict(qid=data[ci]['sample_id'] + ':' + str(qi), method=method, category=q['category'],
                    question=q['question'], hypothesis=result['text'].strip(), answer=q.get('answer'),
                    f1=metric(result['text'].strip(), q), context_tokens=context['tokens'],
                    evidence_recall=len(gold & recalled) / len(gold) if gold else None,
                    all_evidence=gold <= recalled if gold else None, source_ids=context['source_ids'],
                    fact_ids=context['facts'], api_hash=result['request_hash'], usage=result['usage'], state_reader=state)

    with ThreadPoolExecutor(max_workers=args.workers) as pool, predfile.open('a') as out:
        jobs = [pool.submit(answer_one,ci,qi,q,m) for ci,qi,q in chosen for m in methods
                if (data[ci]['sample_id'] + ':' + str(qi),m) not in completed]
        for job in as_completed(jobs):
            r = job.result()
            records.append(r)
            out.write(json.dumps(r,ensure_ascii=False) + '\n'); out.flush()
            print(len(records), '/',len(chosen)*len(methods),r['qid'],r['method'],round(r['f1'],3),flush=True)
    summary = summarize(records)
    atomic_json(run / 'summary.json', summary)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
