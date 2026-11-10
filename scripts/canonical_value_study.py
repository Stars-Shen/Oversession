"""Test whether a weaker local reader benefits from existing teacher-written facts."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import numpy as np
import tiktoken
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from api import ROOT,atomic_json,digest
from canonical_values import retrieve
from calendar_memory import TIME_QUERY
from experiment import scorer,summarize
from memory import Memory


def prepare(folder,baseline_run,temporal_only):
    base=ROOT/'runs/phase3'/baseline_run
    inputs=[json.loads(l) for l in (base/'inputs.jsonl').read_text().splitlines()]
    config=json.loads((base/'config.json').read_text());cache={};rows=[]
    os.environ['TIKTOKEN_CACHE_DIR']=str(ROOT/'runs/tokenizer_cache')
    encoder=SimpleNamespace(encode=tiktoken.get_encoding('cl100k_base').encode_ordinary)
    for row in inputs:
        if temporal_only and not TIME_QUERY.search(row['question']):continue
        name,qi=row['qid'].rsplit(':',1);qi=int(qi)
        if name not in cache:
            cache[name]=(Memory(json.loads((ROOT/'runs/memories_captions'/(name+'.json')).read_text())),
                         np.load(ROOT/'runs/embeddings_captions'/(name+'.npz')))
        memory,emb=cache[name];vector=emb['queries'][qi].astype(np.float32)
        dense={key:sorted(enumerate(emb[key].astype(np.float32)@vector),key=lambda item:(-item[1],item[0])) for key in ('raw','facts')}
        ctx=retrieve(memory,row['question'],3000,encoder,dense)
        rows.append(dict(row,context=ctx['text'],context_tokens=ctx['tokens'],source_ids=ctx['source_ids'],
                         original_source_ids=ctx['original_source_ids'],fact_ids=ctx['fact_ids']))
    config.update(methods=['canonical'],inputs_hash=digest(rows),baseline_run=baseline_run,
                  question_ids=[r['qid'] for r in rows],temporal_only=temporal_only,
                  question_rule=TIME_QUERY.pattern if temporal_only else None,
                  intervention='same four-view source priority; canonical teacher facts replace represented source text; raw fallback for sources with no extracted facts',
                  evidence_recall_definition='provenance-root coverage, not visibility of complete original turns',
                  decision='One fixed value-layer test; no fact-count or mixing-weight sweep.')
    cp=folder/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Run config changed')
    atomic_json(cp,config);(folder/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print('prepared',len(rows),'canonical-value inputs; no writer calls')


def score(folder):
    config=json.loads((folder/'config.json').read_text())
    updates=[json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()]
    assert len(updates)==len(config['question_ids']) and {r['qid'] for r in updates}==set(config['question_ids'])
    base={r['qid']:r for r in json.loads((ROOT/'runs/phase3'/config['baseline_run']/'scored.json').read_text()) if r['method']=='plain'}
    samples={s['sample_id']:s for s in json.loads((ROOT/'data/locomo10.json').read_text())};metric=scorer()
    rows=[];delta=[]
    changed={r['qid']:r for r in updates}
    for before in base.values():
        after=dict(changed.get(before['qid'],before),method='canonical')
        name,qi=after['qid'].rsplit(':',1);q=samples[name]['qa'][int(qi)]
        gold={mid for e in q.get('evidence',[]) for mid in re.findall(r'D\d+:\d+',str(e))}
        after.update(category=q['category'],f1=metric(after['hypothesis'],q),
                     evidence_recall=len(gold&set(after['source_ids']))/len(gold) if gold else None)
        assert after['context_tokens']<=3000
        rows.extend([before,after]);delta.append(after['f1']-before['f1'])
    delta=np.array(delta)
    rng=np.random.default_rng(20261111)
    ci=np.quantile([rng.choice(delta,len(delta),replace=True).mean() for _ in range(10000)],[.025,.975])
    result=dict(complete=True,n=len(base),new_generations=len(updates),reused_outputs=len(base)-len(updates),
                selection_scope=('post-hoc development selection' if config.get('temporal_only') and config['split']=='development'
                                 else 'separate-conversation evaluation; these conversations were used in earlier research revisions'
                                 if config['split']=='evaluation' else 'development'),
                metrics=summarize(rows),delta=float(delta.mean()),bootstrap_ci=ci.tolist(),
                wins=int((delta>0).sum()),losses=int((delta<0).sum()),ties=int((delta==0).sum()),
                truncated_outputs=[r['qid'] for r in updates if not r['stopped_with_eos']],config=config)
    atomic_json(folder/'scored.json',rows);atomic_json(folder/'summary.json',result)
    atomic_json(ROOT/'reports'/(folder.name+'.json'),result)
    print(json.dumps({k:result[k] for k in ['n','delta','wins','losses','ties','truncated_outputs']},indent=2))
    print({m:s['f1'] for m,s in result['metrics'].items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','score']);p.add_argument('--run',default='local_canonical_values_dev')
    p.add_argument('--baseline-run',default='local_cad_greedy_dev');p.add_argument('--temporal-only',action='store_true')
    args=p.parse_args();folder=ROOT/'runs/phase3'/args.run;folder.mkdir(parents=True,exist_ok=True)
    prepare(folder,args.baseline_run,args.temporal_only) if args.command=='prepare' else score(folder)


if __name__=='__main__':main()
