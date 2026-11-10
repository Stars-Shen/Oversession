"""Controlled phase-3 LoCoMo reads with a fixed v2 memory and SiliconFlow only."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from types import SimpleNamespace
import numpy as np
import tiktoken
from api import ROOT,atomic_json,digest
from experiment import ANSWER,COMPACT,choose_questions,scorer,summarize
from memory import Memory
from siliconflow import SiliconFlowAPI,SiliconFlowBalanceError


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--convs',default='0,1')
    p.add_argument('--limit',type=int,default=80)
    p.add_argument('--methods',default='v2')
    p.add_argument('--budget',type=int,default=3000)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--run',default='sf_baseline_dev')
    p.add_argument('--prepare',action='store_true')
    args=p.parse_args()
    data_path=ROOT/'data/locomo10.json';data=json.loads(data_path.read_text())
    convs=[int(x) for x in args.convs.split(',')]
    chosen=choose_questions(data,convs,args.limit,20261111)
    methods=args.methods.split(',');api=SiliconFlowAPI();deliberate_api=SiliconFlowAPI(enable_thinking=True)
    memories={};embeddings={};memory_hashes={};indexes={}
    for ci in convs:
        name=data[ci]['sample_id'];path=ROOT/'runs/memories_captions'/(name+'.json')
        memories[ci]=Memory(json.loads(path.read_text()))
        embeddings[ci]=np.load(ROOT/'runs/embeddings_captions'/(name+'.npz'))
        memory_hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        if 'graph' in methods:
            from semantic_index import SemanticIndex
            indexes[ci]=SemanticIndex(name,memories[ci])
    config=dict(benchmark='LoCoMo categories 1-4, official captions',dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                question_ids=[data[ci]['sample_id']+':'+str(qi) for ci,qi,q in chosen],methods=methods,budget=args.budget,
                answer_model=api.model,provider='SiliconFlow',temperature=0,enable_thinking=False,
                answer_prompt_hash=digest(ANSWER+COMPACT),memory_writer='frozen v2 gpt-6-luna; no new OpenAI calls',
                memory_sha256=memory_hashes,split='development' if set(convs)<={0,1} else 'evaluation')
    if 'deliberate' in methods:config['enable_thinking']={m:m=='deliberate' for m in methods}
    if 'graph' in methods:config['semantic_index_signature']={data[ci]['sample_id']:indexes[ci].signature for ci in convs}
    folder=ROOT/'runs/phase3'/args.run;folder.mkdir(parents=True,exist_ok=True)
    cp=folder/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Run config changed; choose a new run name')
    atomic_json(cp,config)
    if args.prepare:return
    os.environ['TIKTOKEN_CACHE_DIR']=str(ROOT/'runs/tokenizer_cache')
    encoder=SimpleNamespace(encode=tiktoken.get_encoding('cl100k_base').encode_ordinary)
    metric=scorer();pf=folder/'predictions.jsonl'
    records=[json.loads(x) for x in pf.read_text().splitlines()] if pf.exists() else []
    done={(r['qid'],r['method']) for r in records}

    def work(ci,qi,q,method):
        memory=memories[ci];emb=embeddings[ci]
        vector=emb['queries'][qi].astype(np.float32)
        dense={key:sorted(enumerate(emb[key].astype(np.float32)@vector),key=lambda item:(-item[1],item[0])) for key in ('raw','facts')}
        if method in ('v2','deliberate'):context=memory.retrieve(q['question'],'hybrid_packed',args.budget,encoder,dense)
        elif method=='coverage':
            from coverage_memory import retrieve_coverage
            context=retrieve_coverage(memory,q['question'],args.budget,encoder,dense)
        elif method=='table':
            from evidence_table import retrieve_table
            context=retrieve_table(memory,q['question'],args.budget,encoder,dense,api)
        elif method=='graph':context=indexes[ci].retrieve(memory,q['question'],args.budget,encoder,dense,vector)
        elif method=='cross_rank':
            from cross_rank import retrieve_ranked
            rank_path=ROOT/'runs/phase3/cross_rank'/(data[ci]['sample_id']+'_'+str(qi)+'.json')
            context=retrieve_ranked(memory,json.loads(rank_path.read_text())['ranking'],args.budget,encoder)
        else:raise ValueError(method)
        assert context['tokens']<=args.budget
        reader=deliberate_api if method=='deliberate' else api
        result=reader.call(ANSWER+COMPACT,'Memories:\n'+context['text']+'\n\nQuestion: '+q['question'],
                           max_tokens=4096 if method=='deliberate' else 768)
        import re
        gold={mid for e in q.get('evidence',[]) for mid in re.findall(r'D\d+:\d+',str(e))}
        row=dict(qid=data[ci]['sample_id']+':'+str(qi),method=method,category=q['category'],question=q['question'],
                    hypothesis=result['text'],f1=metric(result['text'],q),context_tokens=context['tokens'],
                    evidence_recall=len(gold&set(context['source_ids']))/len(gold) if gold else None,
                    source_ids=context['source_ids'],api_hash=result['request_hash'],usage=result['usage'])
        row['enable_thinking']=reader.enable_thinking
        if 'compiler' in context:
            row['compiler']=context['compiler'];row['compiler_input_tokens']=context['compiler_input_tokens']
            row['original_source_ids']=context['original_source_ids']
        return row

    errors=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool,pf.open('a') as out:
        jobs={pool.submit(work,ci,qi,q,m):(ci,qi,m) for ci,qi,q in chosen for m in methods
              if (data[ci]['sample_id']+':'+str(qi),m) not in done}
        for job in as_completed(jobs):
            try:r=job.result()
            except Exception as error:
                errors.append(dict(case=jobs[job],error=str(error)));print('failed',jobs[job],str(error)[:200],flush=True)
                if isinstance(error,SiliconFlowBalanceError):
                    for pending in jobs:pending.cancel()
                    break
                continue
            records.append(r);out.write(json.dumps(r,ensure_ascii=False)+'\n');out.flush()
            print(len(records),'/',len(chosen)*len(methods),r['qid'],r['method'],round(r['f1'],3),flush=True)
    atomic_json(folder/'errors.json',errors)
    atomic_json(folder/'status.json',dict(expected_records=len(chosen)*len(methods),completed_records=len(records),
                                         complete=len(records)==len(chosen)*len(methods),errors=errors))
    if errors:raise RuntimeError('Some requests failed; successful outputs saved for resume')
    summary=summarize(records);atomic_json(folder/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
