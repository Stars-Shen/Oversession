"""Matched native-thinking experiment on the official Qwen3-4B checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from api import ROOT,atomic_json,digest

MODEL=ROOT/'runs/phase3/models/Qwen3-4B'


def parse_completion(tokens,closing,thinking,decode):
    split=len(tokens)-1-tokens[::-1].index(closing) if closing in tokens else len(tokens)
    final=tokens[split+1:] if thinking else tokens
    answer=decode(final).strip()
    valid=bool(answer) and (not thinking or closing in tokens)
    return dict(hypothesis=answer if valid else 'No information available',
                reasoning=decode(tokens[:split]) if thinking else '',
                reasoning_tokens=split if thinking else 0,closed_reasoning=valid)


def prepare(folder,baseline):
    base=ROOT/'runs/phase3'/baseline
    rows=[json.loads(l) for l in (base/'inputs.jsonl').read_text().splitlines()]
    original=json.loads((base/'config.json').read_text())
    metadata=json.loads((MODEL/'hf_metadata.json').read_text())
    config=dict(model=str(MODEL),model_id='Qwen/Qwen3-4B',revision=metadata['sha'],
                weight_sha256={f['rfilename']:f['lfs']['sha256'] for f in metadata['siblings'] if f['rfilename'].endswith('.safetensors')},
                model_config_sha256=hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest(),
                inputs_hash=digest(rows),question_ids=[r['qid'] for r in rows],
                source_manifest=baseline,memory_sha256=original['memory_sha256'],
                context_budget=3000,writer='frozen v2 Luna',split=original['split'],
                methods=['off','on'],max_new_tokens=dict(off=192,on=4096),
                generation=dict(do_sample=True,temperature=.6,top_p=.95,top_k=20,min_p=0.),
                dtype='bfloat16',seed=20261111,
                decision='One native toggle, no prompt/temperature grid. Require >=3 paired F1 points, on >=3 points above the existing local 30B plain reader, and <=5% incomplete outputs per mode before separate-conversation expansion.',
                interpretation='native reader compute; not a new memory algorithm or a completed DeepSeek run')
    cp=folder/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Run config changed')
    atomic_json(cp,config)
    (folder/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print('prepared',len(rows),'answer-free inputs',flush=True)


def generate(folder,mode,shard,shards):
    import torch
    import transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    config=json.loads((folder/'config.json').read_text())
    rows=[json.loads(l) for l in (folder/'inputs.jsonl').read_text().splitlines()]
    assert digest(rows)==config['inputs_hash'] and 0<=shard<shards
    pf=folder/f'predictions.{mode}.{shard}.jsonl'
    saved=[json.loads(l) for l in pf.read_text().splitlines()] if pf.exists() else []
    done={r['qid'] for r in saved}
    pending=[r for i,r in enumerate(rows) if i%shards==shard and r['qid'] not in done]
    if not pending:print('all shard outputs saved');return
    tok=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    closing=tok.convert_tokens_to_ids('</think>')
    assert closing is not None and tok.convert_ids_to_tokens(closing)=='</think>'
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    atomic_json(folder/f'runtime.{mode}.{shard}.json',dict(torch=torch.__version__,transformers=transformers.__version__,
                gpu=torch.cuda.get_device_name(0),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                mode=mode,shard=shard,shards=shards,closing_token_id=closing))
    with pf.open('a') as out:
        for index,row in enumerate(pending):
            start=time.time();seed=(config['seed']+int(digest(row['qid'])[:8],16))%(2**31)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            messages=[dict(role='system',content=row['system']),
                      dict(role='user',content='Memories:\n'+row['context']+'\n\nQuestion: '+row['question'])]
            ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                        enable_thinking=mode=='on',return_tensors='pt').to('cuda')
            with torch.inference_mode():
                output=model.generate(ids,attention_mask=torch.ones_like(ids),**config['generation'],
                                      max_new_tokens=config['max_new_tokens'][mode],
                                      pad_token_id=tok.pad_token_id,logits_to_keep=1)
            tokens=output[0,ids.shape[1]:].tolist()
            result=dict(qid=row['qid'],method=mode,
                        **parse_completion(tokens,closing,mode=='on',lambda part:tok.decode(part,skip_special_tokens=True)),
                        completion_tokens=len(tokens),input_tokens=ids.shape[1],
                        stopped_with_eos=bool(tokens and tokens[-1] in eos),seed=seed,
                        elapsed_seconds=time.time()-start,context_tokens=row['context_tokens'],source_ids=row['source_ids'])
            out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush()
            print(mode,shard,index+1,'/',len(pending),row['qid'],repr(result['hypothesis']),len(tokens),round(result['elapsed_seconds'],2),flush=True)
    atomic_json(folder/f'status.{mode}.{shard}.json',dict(complete=True,shard=shard,shards=shards,mode=mode))


def score(folder):
    import numpy as np
    import re
    from experiment import scorer,summarize
    config=json.loads((folder/'config.json').read_text())
    rows=[json.loads(l) for p in sorted(folder.glob('predictions.*.jsonl')) for l in p.read_text().splitlines()]
    expected={(qid,m) for qid in config['question_ids'] for m in config['methods']}
    assert len(rows)==len(expected) and {(r['qid'],r['method']) for r in rows}==expected
    inputs={r['qid']:r for r in map(json.loads,(folder/'inputs.jsonl').read_text().splitlines())}
    samples={s['sample_id']:s for s in json.loads((ROOT/'data/locomo10.json').read_text())};metric=scorer()
    for row in rows:
        assert row['source_ids']==inputs[row['qid']]['source_ids'] and row['context_tokens']<=3000
        name,qi=row['qid'].rsplit(':',1);q=samples[name]['qa'][int(qi)]
        gold={mid for e in q.get('evidence',[]) for mid in re.findall(r'D\d+:\d+',str(e))}
        row.update(category=q['category'],f1=metric(row['hypothesis'],q),
                   evidence_recall=len(gold&set(row['source_ids']))/len(gold) if gold else None)
    by={m:{r['qid']:r for r in rows if r['method']==m} for m in config['methods']}
    reference=[r for r in json.loads((ROOT/'runs/phase3'/config['source_manifest']/'scored.json').read_text()) if r['method']=='plain']
    assert {r['qid'] for r in reference}==set(config['question_ids'])
    reference_f1=float(np.mean([r['f1'] for r in reference]))
    on_f1=float(np.mean([r['f1'] for r in by['on'].values()]))
    delta=np.array([by['on'][qid]['f1']-by['off'][qid]['f1'] for qid in config['question_ids']])
    rng=np.random.default_rng(config['seed'])
    ci=np.quantile([rng.choice(delta,len(delta),replace=True).mean() for _ in range(10000)],[.025,.975])
    incomplete=[dict(qid=r['qid'],method=r['method']) for r in rows if not r['stopped_with_eos'] or not r['closed_reasoning']]
    result=dict(complete=True,n=len(delta),metrics=summarize(rows),delta=float(delta.mean()),bootstrap_ci=ci.tolist(),
                wins=int((delta>0).sum()),losses=int((delta<0).sum()),ties=int((delta==0).sum()),incomplete_outputs=incomplete,
                usage={m:{key:sum(r[key] for r in by[m].values()) for key in ('input_tokens','completion_tokens','reasoning_tokens')} for m in config['methods']},
                existing_local_reader_f1=reference_f1,delta_vs_existing_local_reader=on_f1-reference_f1,
                expansion_gate_passed=bool(delta.mean()>=.03 and on_f1-reference_f1>=.03 and all(sum(r['method']==m for r in incomplete)<=.05*len(delta)
                                                                 for m in config['methods'])),config=config)
    atomic_json(folder/'scored.json',rows);atomic_json(folder/'summary.json',result)
    atomic_json(ROOT/'reports'/(folder.name+'.json'),result)
    print(json.dumps({k:result[k] for k in ('n','delta','bootstrap_ci','wins','losses','ties','incomplete_outputs','expansion_gate_passed')},indent=2))
    print({m:s['f1'] for m,s in result['metrics'].items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','generate','score'])
    p.add_argument('--run',default='local_native_capacity_dev');p.add_argument('--baseline-run',default='local_cad_greedy_dev')
    p.add_argument('--mode',choices=['off','on'],default='off');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    args=p.parse_args();folder=ROOT/'runs/phase3'/args.run;folder.mkdir(parents=True,exist_ok=True)
    if args.command=='prepare':prepare(folder,args.baseline_run)
    elif args.command=='generate':generate(folder,args.mode,args.shard,args.shards)
    else:score(folder)


if __name__=='__main__':main()
