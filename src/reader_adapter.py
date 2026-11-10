"""One fixed, dialogue-separated LoRA reader adaptation; frozen memory/retrieval."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
from api import ROOT,atomic_json,digest


def attach(model,rank,alpha):
    import torch
    import torch.nn.functional as F
    class Linear(torch.nn.Module):
        def __init__(self,base):
            super().__init__();self.base=base;self.scale=alpha/rank
            self.a=torch.nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
            self.b=torch.nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
        def forward(self,x):
            update=F.linear(F.linear(x.float(),self.a),self.b)*self.scale
            return self.base(x)+update.to(x.dtype)
    model.requires_grad_(False);layers={}
    for name,module in list(model.named_modules()):
        if name.endswith(('.q_proj','.v_proj')):
            assert isinstance(module,torch.nn.Linear)
            parent,leaf=name.rsplit('.',1);layer=Linear(module)
            setattr(model.get_submodule(parent),leaf,layer);layers[name]=layer
    assert layers
    return layers


def load_adapter(model,path,config):
    import torch
    layers=attach(model,config['rank'],config['alpha'])
    weights=torch.load(path,map_location='cpu',weights_only=True)
    assert set(weights)=={name+'.'+part for name in layers for part in ('a','b')}
    with torch.no_grad():
        for name,layer in layers.items():
            for part in ('a','b'):getattr(layer,part).copy_(weights[name+'.'+part])


def answer_loss(model,ids,answer_length):
    import torch
    import torch.nn.functional as F
    logits=model(ids,attention_mask=torch.ones_like(ids),use_cache=False,
                 logits_to_keep=answer_length+1).logits
    return F.cross_entropy(logits[0,:-1].float(),ids[0,-answer_length:])


def prepare(folder):
    source=ROOT/'runs/phase3/local_reader_adapter_source'
    rows=[json.loads(l) for l in (source/'inputs.jsonl').read_text().splitlines()]
    data=json.loads((ROOT/'data/locomo10.json').read_text());train_name=data[0]['sample_id'];val_name=data[1]['sample_id']
    training=[];validation=[]
    for row in rows:
        name,qi=row['qid'].rsplit(':',1)
        if name==train_name:
            q=data[0]['qa'][int(qi)];answer=str(q['answer'])
            if q['category']==3:answer=answer.split(';')[0]
            training.append(dict(row,target=answer))
        elif name==val_name:validation.append(row)
        else:raise ValueError('Unexpected dialogue in training preparation')
    assert len(training)==152 and len(validation)==81
    assert not {r['qid'] for r in training}&{r['qid'] for r in validation}
    config=dict(base_run='local_native_capacity_dev',train_dialogue=train_name,validation_dialogue=val_name,
                train_n=152,validation_n=81,train_hash=digest(training),validation_hash=digest(validation),
                rank=8,alpha=16,targets=['q_proj','v_proj'],epochs=3,learning_rate=1e-4,
                gradient_accumulation=8,gradient_clip=1.,seed=20261111,
                scope='supervised reader adaptation, not Memory-R1 RL reproduction',
                decision='One fixed training schedule, last adapter only. Validate once on the other dialogue; no epoch or parameter search.')
    cp=folder/'training_config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Training config changed')
    atomic_json(cp,config)
    for name,values in [('training',training),('validation',validation)]:
        (folder/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
    base_config=json.loads((ROOT/'runs/phase3/local_native_capacity_dev/config.json').read_text())
    base_config.update(question_ids=[r['qid'] for r in validation],inputs_hash=digest(validation),methods=['off'],
                       split='dialogue-separated validation',source_manifest='local_reader_adapter_source',
                       decision='Unadapted control for fixed reader-adapter validation; no native-thinking comparison here.',
                       interpretation='Same off-mode sampling and evidence; compare with one final task-local adapter.')
    baseline=ROOT/'runs/phase3/local_adapter_baseline_validation';baseline.mkdir(exist_ok=True)
    atomic_json(baseline/'config.json',base_config)
    (baseline/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in validation))
    old_inputs={r['qid']:r for r in map(json.loads,(ROOT/'runs/phase3/local_native_capacity_dev/inputs.jsonl').read_text().splitlines())}
    old_outputs=[json.loads(l) for l in (ROOT/'runs/phase3/local_native_capacity_dev/predictions.off.0.jsonl').read_text().splitlines()]
    by_qid={r['qid']:r for r in validation};reused=[]
    for row in old_outputs:
        if row['qid'] in by_qid:
            assert by_qid[row['qid']]==old_inputs[row['qid']]
            reused.append(row)
    pf=baseline/'predictions.off.0.jsonl'
    if not pf.exists():pf.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reused))
    print('training',len(training),'validation',len(validation),'reused baseline predictions',len(reused),flush=True)


def train(folder):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    config=json.loads((folder/'training_config.json').read_text())
    rows=[json.loads(l) for l in (folder/'training.jsonl').read_text().splitlines()]
    assert digest(rows)==config['train_hash']
    if (folder/'adapter.pt').exists():raise ValueError('Final adapter already exists; do not retrain')
    base=json.loads((ROOT/'runs/phase3'/config['base_run']/'config.json').read_text())
    torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    tok=AutoTokenizer.from_pretrained(base['model'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(base['model'],local_files_only=True,
                    torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    layers=attach(model,config['rank'],config['alpha']);parameters=[p for p in model.parameters() if p.requires_grad]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
    optimizer=torch.optim.AdamW(parameters,lr=config['learning_rate'],weight_decay=0.)
    examples=[];eos=tok.convert_tokens_to_ids('<|im_end|>')
    for row in rows:
        prompt=tok.apply_chat_template([dict(role='system',content=row['system']),
            dict(role='user',content='Memories:\n'+row['context']+'\n\nQuestion: '+row['question'])],
            tokenize=True,add_generation_prompt=True,enable_thinking=False)
        target=tok.encode(row['target'],add_special_tokens=False)+[eos]
        examples.append((prompt+target,len(target)))
    rng=random.Random(config['seed']);step=0;start=time.time()
    with (folder/'training_log.jsonl').open('a') as log:
        for epoch in range(config['epochs']):
            order=list(range(len(examples)));rng.shuffle(order)
            for offset in range(0,len(order),config['gradient_accumulation']):
                batch=order[offset:offset+config['gradient_accumulation']];optimizer.zero_grad(set_to_none=True);loss_sum=0.
                for index in batch:
                    values,count=examples[index];ids=torch.tensor([values],device='cuda')
                    loss=answer_loss(model,ids,count);loss_sum+=float(loss.detach())
                    (loss/len(batch)).backward()
                norm=torch.nn.utils.clip_grad_norm_(parameters,config['gradient_clip']);optimizer.step();step+=1
                item=dict(epoch=epoch+1,step=step,loss=loss_sum/len(batch),gradient_norm=float(norm),elapsed_seconds=time.time()-start)
                log.write(json.dumps(item)+'\n');log.flush();print(item,flush=True)
    temporary=folder/'adapter.tmp.pt'
    torch.save({name+'.'+part:getattr(layer,part).detach().cpu() for name,layer in layers.items() for part in ('a','b')},temporary)
    temporary.replace(folder/'adapter.pt')
    atomic_json(folder/'training_status.json',dict(complete=True,steps=step,epochs=config['epochs'],
                adapter_sha256=hashlib.sha256((folder/'adapter.pt').read_bytes()).hexdigest(),
                trainable_parameters=sum(p.numel() for p in parameters),torch=torch.__version__,transformers=transformers.__version__,
                max_cuda_memory_gib=torch.cuda.max_memory_allocated()/2**30,elapsed_seconds=time.time()-start))


def prepare_evaluation(folder):
    status=json.loads((folder/'training_status.json').read_text());training=json.loads((folder/'training_config.json').read_text())
    assert status['complete'] and status['epochs']==training['epochs']
    base=ROOT/'runs/phase3/local_adapter_baseline_validation';target=ROOT/'runs/phase3/local_adapter_validation'
    target.mkdir(exist_ok=True);config=json.loads((base/'config.json').read_text())
    config.update(adapter_file=str((folder/'adapter.pt').relative_to(ROOT)),adapter_sha256=status['adapter_sha256'],
                  adapter_config={key:training[key] for key in ('rank','alpha')},
                  decision='One final three-epoch adapter, one validation. Require >=3 paired F1 points and <=5% incomplete outputs per condition before separate-conversation evaluation.')
    cp=target/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Adapter evaluation config changed')
    atomic_json(cp,config);(target/'inputs.jsonl').write_text((base/'inputs.jsonl').read_text())
    print('prepared final-adapter validation',len(config['question_ids']),flush=True)


def score(folder):
    import numpy as np
    import re
    from experiment import scorer,summarize
    target=ROOT/'runs/phase3/local_adapter_validation';baseline=ROOT/'runs/phase3/local_adapter_baseline_validation'
    config=json.loads((target/'config.json').read_text());ids=set(config['question_ids'])
    samples={s['sample_id']:s for s in json.loads((ROOT/'data/locomo10.json').read_text())};metric=scorer()
    combined=[];by={}
    for method,run in [('base',baseline),('adapted',target)]:
        rows=[json.loads(l) for p in sorted(run.glob('predictions.off.*.jsonl')) for l in p.read_text().splitlines()]
        assert len(rows)==len(ids) and {r['qid'] for r in rows}==ids
        for row in rows:
            name,qi=row['qid'].rsplit(':',1);q=samples[name]['qa'][int(qi)]
            gold={mid for e in q.get('evidence',[]) for mid in re.findall(r'D\d+:\d+',str(e))}
            row.update(method=method,category=q['category'],f1=metric(row['hypothesis'],q),
                       evidence_recall=len(gold&set(row['source_ids']))/len(gold) if gold else None)
            assert row['context_tokens']<=3000
        by[method]={r['qid']:r for r in rows};combined.extend(rows)
    for qid in ids:assert by['base'][qid]['source_ids']==by['adapted'][qid]['source_ids']
    delta=np.array([by['adapted'][qid]['f1']-by['base'][qid]['f1'] for qid in sorted(ids)])
    rng=np.random.default_rng(20261111)
    ci=np.quantile([rng.choice(delta,len(delta),replace=True).mean() for _ in range(10000)],[.025,.975])
    incomplete=[dict(qid=r['qid'],method=r['method']) for r in combined if not r['stopped_with_eos'] or not r['closed_reasoning']]
    result=dict(complete=True,n=len(ids),scope='one heldout dialogue; not a full benchmark or multi-user confirmation',
                metrics=summarize(combined),delta=float(delta.mean()),bootstrap_ci=ci.tolist(),
                wins=int((delta>0).sum()),losses=int((delta<0).sum()),ties=int((delta==0).sum()),
                incomplete_outputs=incomplete,training=json.loads((folder/'training_config.json').read_text()),
                training_status=json.loads((folder/'training_status.json').read_text()),config=config,
                expansion_gate_passed=bool(delta.mean()>=.03 and all(sum(r['method']==m for r in incomplete)<=.05*len(ids) for m in by)))
    atomic_json(target/'scored.json',combined);atomic_json(target/'summary.json',result)
    atomic_json(ROOT/'reports/reader_adapter_validation.json',result)
    print(json.dumps({k:result[k] for k in ('n','delta','bootstrap_ci','wins','losses','ties','incomplete_outputs','expansion_gate_passed')},indent=2))
    print({m:s['f1'] for m,s in result['metrics'].items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','prepare-evaluation','score'])
    args=p.parse_args();folder=ROOT/'runs/phase3/reader_adapter';folder.mkdir(parents=True,exist_ok=True)
    {'prepare':prepare,'train':train,'prepare-evaluation':prepare_evaluation,'score':score}[args.command](folder)


if __name__=='__main__':main()
