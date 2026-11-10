"""LongMemEval-S adapter. Label fields are isolated from the memory writer."""
import argparse
import ast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from types import SimpleNamespace
import numpy as np
import tiktoken
from api import API, ROOT, atomic_json, digest
from memory import Memory, extract_session, evolve
from experiment import ANSWER
from state_reader import compile_state, PROMPT as STATE_PROMPT


def sessions_for(entry):
    sessions=[]
    # IDs are opaque local positions; original IDs can contain answer/noans prefixes.
    for i,(date,turns) in enumerate(zip(entry['haystack_dates'],entry['haystack_sessions'])):
        sid='S'+str(i)
        clean=[dict(id=sid+':T'+str(j),speaker=t['role'],text=t['content'],date=date,session=sid,order=i)
               for j,t in enumerate(turns)]
        if clean:
            sessions.append(dict(id=sid,date=date,turns=clean))
    return sessions


def select(data, per_type):
    groups=defaultdict(list)
    for e in data:
        groups[e['question_type']].append(e)
    chosen=[]
    for kind in sorted(groups):
        chosen.extend(sorted(groups[kind],key=lambda e:digest(e['question_id']))[:per_type])
    return chosen


def judge_prompt():
    p=ROOT/'third_party/LongMemEval/src/evaluation/evaluate_qa.py'
    tree=ast.parse(p.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_anscheck_prompt']
    env={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'official_longmem_judge_prompt','exec'),env)
    return env['get_anscheck_prompt']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['build','evaluate','recall'])
    p.add_argument('--per-type',type=int,default=4)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--budget',type=int,default=6000)
    p.add_argument('--methods',default='raw,facts')
    p.add_argument('--run',default='longmem_dev_v1')
    p.add_argument('--split',choices=['development','validation','full'],default='development')
    p.add_argument('--state-reader',action='store_true')
    p.add_argument('--ready-only',action='store_true',help='Schedule available memories; keep the full frozen evaluation scope')
    args=p.parse_args()
    path=ROOT/'data/longmemeval_s_cleaned.json'
    data=json.loads(path.read_text())
    if args.split=='validation':
        dev_ids=set(json.loads((ROOT/'runs/longmem_dev_ids.json').read_text()))
        data=[e for e in data if e['question_id'] not in dev_ids]
    chosen=data if args.split=='full' else select(data,args.per_type)
    run=ROOT/'runs'/args.run;run.mkdir(parents=True,exist_ok=True)
    split=ROOT/'runs'/({'development':'longmem_dev_ids.json','validation':'longmem_validation_ids.json',
                       'full':'longmem_full_ids.json'}[args.split])
    selected_ids=[e['question_id'] for e in chosen]
    if split.exists() and json.loads(split.read_text()) != selected_ids:
        raise ValueError('Development split already frozen with different IDs')
    atomic_json(split,selected_ids)
    api=API()
    if args.command=='build':
        all_sessions={e['question_id']:sessions_for(e) for e in chosen}
        extracted={qid:{} for qid in all_sessions}
        errors=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs={pool.submit(extract_session,s,api):(qid,s['id']) for qid,ss in all_sessions.items() for s in ss}
            done=0
            for job in as_completed(jobs):
                qid,sid=jobs[job]
                try:
                    extracted[qid][sid]=job.result()
                except Exception as error:
                    errors.append(dict(qid=qid,session=sid,error=str(error)))
                    extracted[qid][sid]=dict(facts=[],error=str(error))
                    print('extraction failed',qid,sid,str(error)[:150],flush=True)
                done+=1
                if done%10==0:print('sessions',done,'/',len(jobs),flush=True)
        for qid,ss in all_sessions.items():
            facts=[f for s in ss for f in extracted[qid][s['id']]['facts']]
            memory=dict(sessions=ss,facts=evolve(facts),extraction=extracted[qid],schema='vem-0.1')
            atomic_json(ROOT/'runs/longmem_memories'/(qid+'.json'),memory)
            print('built',qid,len(facts),'state edges',sum(len(f['previous']) for f in memory['facts']),flush=True)
        atomic_json(run/'extraction_errors.json',errors)
        return
    os.environ['TIKTOKEN_CACHE_DIR']=str(ROOT/'runs/tokenizer_cache')
    encoder=SimpleNamespace(encode=tiktoken.get_encoding('cl100k_base').encode_ordinary)
    methods=args.methods.split(',')
    config=dict(dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),question_ids=selected_ids,
                methods=methods,budget=args.budget,model=api.model,split=args.split,
                judge_model=api.model,judge_protocol='unchanged official LongMemEval prompts, substitute Luna judge; NOT official GPT-4o leaderboard score',
                answer_prompt_hash=digest(ANSWER),tokenizer='cl100k_base')
    if args.state_reader:
        config['state_reader_prompt_hash']=digest(STATE_PROMPT)
        config['state_reader_budget']='same total final evidence budget; one extra compiler call'
    if any('hybrid' in m for m in methods):
        config['embedding_model']='Qwen3-Embedding-0.6B local, float16, normalized, max_seq_length=2048'
    cp=run/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Choose a new run name')
    atomic_json(cp,config)
    records=[];pf=run/'predictions.jsonl'
    if pf.exists():records=[json.loads(x) for x in pf.read_text().splitlines()]
    completed={(r['qid'],r['method']) for r in records}
    judge=judge_prompt()

    def work(e,method):
        if method in ('raw','hybrid'):
            memory=Memory(dict(sessions=sessions_for(e),facts=[]))
        else:
            memory_dir='longmem_consolidated' if method.startswith('consolidated') else 'longmem_memories'
            memory=Memory(json.loads((ROOT/'runs'/memory_dir/(e['question_id']+'.json')).read_text()))
        dense=None
        if 'hybrid' in method:
            emb=np.load(ROOT/'runs/longmem_embeddings'/(e['question_id']+'.npz'))
            if list(emb['document_ids'])!=[t['id'] for t in memory.turns]:raise ValueError('Embedding source IDs do not match')
            similarities=emb['raw'].astype(np.float32)@emb['query'].astype(np.float32)
            dense=dict(raw=sorted(enumerate(similarities),key=lambda item:(-item[1],item[0])),facts=[])
        ctx=memory.retrieve(e['question'],method,args.budget,encoder,dense)
        state=None
        if args.state_reader:
            state=compile_state(api,e['question'],e['question_date'],ctx,args.budget,encoder)
            ctx=memory.retrieve(e['question'],method,args.budget-state['tokens']-32,encoder,dense)
            state_sources=set(ctx['source_ids'])
            for observation in json.loads(state['text']).get('observations',[]):
                for reference in observation['sources']:
                    state_sources.update(memory.by_fact[reference]['sources'] if reference in memory.by_fact else [reference])
            ctx['source_ids']=sorted(state_sources)
            ctx['tokens']+=state['tokens']
        # Gold evidence is accessed only after retrieval; never passed to the writer or answerer.
        recovered={int(t.split(':')[0][1:]) for t in ctx['source_ids']}
        gold={i for i,sid in enumerate(e['haystack_session_ids']) if sid in e['answer_session_ids']}
        base=dict(qid=e['question_id'],method=method,question_type=e['question_type'],
                  evidence_session_recall=len(recovered&gold)/len(gold) if gold else None,
                  all_evidence=gold<=recovered if gold else None,context_tokens=ctx['tokens'],source_ids=ctx['source_ids'])
        if args.command=='recall':return base
        state_text=('\nCompiled evidence state (advisory; original turns are authoritative):\n'+state['text']) if state else ''
        response=api.call(ANSWER,'Current date: '+e['question_date']+'\nMemories:\n'+ctx['text']+state_text+'\n\nQuestion: '+e['question'],max_tokens=1536)
        prompt=judge(e['question_type'],e['question'],e['answer'],response['text'],abstention=e['question_id'].endswith('_abs'))
        evaluation=api.call('Evaluate using the rubric. Return yes or no only.',prompt,max_tokens=256)
        value=evaluation['text'].strip().lower().rstrip('.')
        if value not in ('yes','no'):raise ValueError('Invalid judge output: '+value)
        return dict(base,hypothesis=response['text'],correct=value=='yes',answer_api_hash=response['request_hash'],
                    judge_api_hash=evaluation['request_hash'],answer_usage=response['usage'],
                    state_reader=state)

    def ready(e,method):
        if not args.ready_only:return True
        if 'hybrid' in method and not (ROOT/'runs/longmem_embeddings'/(e['question_id']+'.npz')).exists():return False
        if method in ('raw','hybrid'):return True
        memory_dir='longmem_consolidated' if method.startswith('consolidated') else 'longmem_memories'
        return (ROOT/'runs'/memory_dir/(e['question_id']+'.json')).exists()

    with ThreadPoolExecutor(max_workers=args.workers) as pool,pf.open('a') as out:
        jobs=[pool.submit(work,e,m) for e in chosen for m in methods if (e['question_id'],m) not in completed and ready(e,m)]
        for job in as_completed(jobs):
            row=job.result();records.append(row);out.write(json.dumps(row)+'\n');out.flush()
            print(len(records),'/',len(chosen)*len(methods),row['qid'],row['method'],row.get('correct'),flush=True)
    summary={}
    for m in methods:
        rows=[r for r in records if r['method']==m]
        summary[m]=dict(n=len(rows),accuracy=float(np.mean([r['correct'] for r in rows])) if args.command=='evaluate' else None,
                        evidence_session_recall=float(np.mean([r['evidence_session_recall'] for r in rows if r['evidence_session_recall'] is not None])),
                        by_type={k:dict(n=sum(r['question_type']==k for r in rows),
                                      accuracy=float(np.mean([r['correct'] for r in rows if r['question_type']==k])) if args.command=='evaluate' else None)
                                 for k in sorted({r['question_type'] for r in rows})})
        if args.ready_only:
            summary[m]['expected_n']=len(chosen)
            summary[m]['complete']=len(rows)==len(chosen)
    atomic_json(run/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
