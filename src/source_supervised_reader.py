"""One fixed 30B reader update with training-only positive-source supervision."""
import argparse
import hashlib
import json
import os
import random
import re
import time
from api import ROOT,atomic_json,digest


def read_rows(path):return [json.loads(line) for line in path.read_text().splitlines()]


def save_rows(path,rows):
    text=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
    if path.exists() and path.read_text()!=text:raise ValueError('Manifest changed: '+str(path))
    path.write_text(text)


def prepare(folder):
    data=json.loads((ROOT/'data/locomo10.json').read_text())
    rows=read_rows(ROOT/'runs/phase3/local_reader_adapter_source/inputs.jsonl')
    train=[];validation=[]
    for row in rows:
        name,qi=row['qid'].rsplit(':',1)
        if name==data[1]['sample_id']:validation.append(row);continue
        assert name==data[0]['sample_id']
        q=data[0]['qa'][int(qi)]
        sources={s for e in q.get('evidence',[]) for s in re.findall(r'D\d+:\d+',str(e))}
        if not sources or not sources<=set(row['source_ids']):continue
        target=str(q['answer'])
        if q['category']==3:target=target.split(';')[0]
        train.append(dict(row,target=target,positive_sources=sorted(sources)))
    assert len(train)==131 and len(validation)==81
    config=dict(base_run='local_cad_greedy_dev',train_dialogue=data[0]['sample_id'],
        validation_dialogue=data[1]['sample_id'],train_n=len(train),validation_n=len(validation),
        train_hash=digest(train),validation_hash=digest(validation),rank=8,alpha=16,
        targets=['q_proj','v_proj'],epochs=3,learning_rate=1e-4,gradient_accumulation=8,
        gradient_clip=1.,seed=20261111,source_loss_weight=.1,
        answer_loss='token-normalized within each accumulated batch, including EOS',
        source_loss='mean positive-source negative log of maximum head content mass at last prompt token',
        decision='Fixed final adapter only; no epoch, rank or weight search. Require +3 paired F1 points and <=5% capped outputs on validation before separate-conversation evaluation.',
        interpretation='Joint supervised-reader update; does not isolate attention regularization from answer supervision or token normalization. Attention is not a faithfulness certificate.')
    cp=folder/'training_config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Training config changed')
    atomic_json(cp,config);save_rows(folder/'training.jsonl',train);save_rows(folder/'validation.jsonl',validation)
    base=json.loads((ROOT/'runs/phase3/local_cad_greedy_dev/config.json').read_text())
    base.update(methods=['plain'],question_ids=[r['qid'] for r in validation],inputs_hash=digest(validation),
                split='conversation 1 validation for source-supervised reader',
                source_manifest='local_reader_adapter_source')
    run=ROOT/'runs/phase3/local_source_control_validation';run.mkdir(exist_ok=True)
    atomic_json(run/'config.json',base);save_rows(run/'inputs.jsonl',validation)
    old=ROOT/'runs/phase3/local_cad_greedy_dev'
    inputs={r['qid']:r for r in read_rows(old/'inputs.jsonl')};val={r['qid']:r for r in validation}
    reused=[]
    for row in read_rows(old/'predictions.jsonl'):
        if row['method']=='plain' and row['qid'] in val:
            assert inputs[row['qid']]==val[row['qid']];reused.append(row)
    if not (run/'predictions.jsonl').exists():save_rows(run/'predictions.jsonl',reused)
    print(dict(train=len(train),multisource=sum(len(r['positive_sources'])>1 for r in train),
               validation=len(validation),reused_controls=len(reused)),flush=True)


def train(folder):
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    from local_cad import active_experts_forward
    from reader_adapter import attach
    from source_attention import SourceAttention,coverage_loss,encoded_prompt
    config=json.loads((folder/'training_config.json').read_text())
    rows=read_rows(folder/'training.jsonl');assert digest(rows)==config['train_hash']
    if (folder/'adapter.pt').exists():raise ValueError('Final adapter exists; do not retrain')
    base=json.loads((ROOT/'runs/phase3'/config['base_run']/'config.json').read_text())
    torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    tok=AutoTokenizer.from_pretrained(base['model'],local_files_only=True)
    examples=[];eos=tok.convert_tokens_to_ids('<|im_end|>')
    for row in rows:
        prompt,spans=encoded_prompt(tok,row)
        expected=tok.apply_chat_template([dict(role='system',content=row['system']),
            dict(role='user',content='Memories:\n'+row['context']+'\n\nQuestion: '+row['question'])],
            tokenize=True,add_generation_prompt=True)
        assert prompt==expected
        target=tok.encode(row['target'],add_special_tokens=False)+[eos]
        examples.append((prompt+target,len(prompt),len(target),spans,set(row['positive_sources'])))
    Qwen3MoeSparseMoeBlock.forward=active_experts_forward
    model=AutoModelForCausalLM.from_pretrained(base['model'],local_files_only=True,
        torch_dtype=torch.bfloat16,device_map='balanced',
        max_memory={i:'22GiB' for i in range(torch.cuda.device_count())},attn_implementation='sdpa')
    assert all(str(d) not in ('cpu','disk') for d in model.hf_device_map.values())
    layers=attach(model,config['rank'],config['alpha']);parameters=[p for p in model.parameters() if p.requires_grad]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
    capture=SourceAttention(model);device=model.get_input_embeddings().weight.device
    optimizer=torch.optim.AdamW(parameters,lr=config['learning_rate'],weight_decay=0.)
    def losses(example):
        values,prompt_length,count,spans,positive=example
        ids=torch.tensor([values],device=device)
        logits=model(ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=count+1).logits
        ce=F.cross_entropy(logits[0,:-1].float(),ids[0,-count:].to(logits.device))
        coverage=coverage_loss(capture.attention(prompt_length-1),spans,positive)
        return ce,coverage.to(ce.device)
    # One real maximum-length backward pass, no optimizer step; then clear it.
    start=time.time();ce,coverage=losses(max(examples,key=lambda x:len(x[0])))
    (ce+config['source_loss_weight']*coverage).backward()
    norm=torch.nn.utils.clip_grad_norm_(parameters,config['gradient_clip'])
    assert torch.isfinite(norm) and norm>0
    atomic_json(folder/'smoke.json',dict(finite_backward=True,optimizer_steps=0,
        answer_loss=float(ce.detach()),source_loss=float(coverage.detach()),gradient_norm=float(norm),
        max_input_tokens=max(len(x[0]) for x in examples),elapsed_seconds=time.time()-start))
    optimizer.zero_grad(set_to_none=True);del ce,coverage,norm
    print('maximum-length forward/backward smoke passed',flush=True)
    rng=random.Random(config['seed']);step=0;start=time.time()
    with (folder/'training_log.jsonl').open('a') as log:
        for epoch in range(config['epochs']):
            order=list(range(len(examples)));rng.shuffle(order)
            for offset in range(0,len(order),config['gradient_accumulation']):
                batch=order[offset:offset+config['gradient_accumulation']]
                total_tokens=sum(examples[i][2] for i in batch);optimizer.zero_grad(set_to_none=True)
                ce_sum=0.;coverage_sum=0.
                for i in batch:
                    ce,coverage=losses(examples[i]);weight=examples[i][2]/total_tokens
                    loss=ce*weight+config['source_loss_weight']*coverage/len(batch)
                    ce_sum+=float(ce.detach())*weight;coverage_sum+=float(coverage.detach())/len(batch)
                    loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(parameters,config['gradient_clip'])
                assert torch.isfinite(norm)
                optimizer.step();step+=1
                item=dict(epoch=epoch+1,step=step,answer_loss=ce_sum,source_loss=coverage_sum,
                          gradient_norm=float(norm),elapsed_seconds=time.time()-start)
                log.write(json.dumps(item)+'\n');log.flush();print(item,flush=True)
    capture.close()
    temporary=folder/'adapter.tmp.pt'
    torch.save({name+'.'+part:getattr(layer,part).detach().cpu() for name,layer in layers.items()
                for part in ('a','b')},temporary);temporary.replace(folder/'adapter.pt')
    atomic_json(folder/'training_status.json',dict(complete=True,steps=step,epochs=config['epochs'],
        adapter_sha256=hashlib.sha256((folder/'adapter.pt').read_bytes()).hexdigest(),
        trainable_parameters=sum(p.numel() for p in parameters),torch=torch.__version__,transformers=transformers.__version__,
        max_cuda_memory_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())],
        elapsed_seconds=time.time()-start,device_map={k:str(v) for k,v in model.hf_device_map.items()},
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))


def prepare_evaluation(folder,separate=False):
    status=json.loads((folder/'training_status.json').read_text());assert status['complete']
    training=json.loads((folder/'training_config.json').read_text())
    source=ROOT/'runs/phase3'/('local_plain_validation' if separate else 'local_source_control_validation')
    target=ROOT/'runs/phase3'/('local_source_separate' if separate else 'local_source_validation');target.mkdir(exist_ok=True)
    config=json.loads((source/'config.json').read_text())
    config.update(adapter_file=str((folder/'adapter.pt').relative_to(ROOT)),adapter_sha256=status['adapter_sha256'],
                  adapter_config={k:training[k] for k in ('rank','alpha')})
    if separate:
        validation=json.loads((ROOT/'reports/source_reader_validation.json').read_text())
        assert validation['training_status']['adapter_sha256']==status['adapter_sha256']
        config['triage_override']=dict(original_gate_passed=validation['expansion_gate_passed'],
            validation_delta=validation['delta'],
            reason='One explicit final-adapter confirmation: +2.58 on one dialogue, positive in all three observed categories, no inference questions. No retraining or tuning.',
            full_run_decision='Require >=3 paired F1 points and >=3 target-mixture sensitivity points, with <=5% capped outputs. No further exception for this candidate.')
    cp=target/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Evaluation config changed')
    atomic_json(cp,config);save_rows(target/'inputs.jsonl',read_rows(source/'inputs.jsonl'))


def prepare_full(folder):
    from types import SimpleNamespace
    from local_cad import prepare as prepare_inputs
    report=json.loads((ROOT/'reports/source_reader_separate.json').read_text())
    assert report['expansion_gate_passed'] and report['target_mix_sensitivity']['delta']>=.03
    status=json.loads((folder/'training_status.json').read_text());assert status['complete']
    assert status['adapter_sha256']==report['training_status']['adapter_sha256']
    training=json.loads((folder/'training_config.json').read_text())
    baseline=ROOT/'runs/phase3/local_source_control_full';baseline.mkdir(exist_ok=True)
    if not (baseline/'inputs.jsonl').exists():
        prepare_inputs(SimpleNamespace(convs='2,3,4,5,6,7,8,9',limit=0,plain_only=True),baseline)
    rows=read_rows(baseline/'inputs.jsonl');by_id={r['qid']:r for r in rows}
    assert len(rows)==len(by_id)==1307
    config=json.loads((baseline/'config.json').read_text());assert digest(rows)==config['inputs_hash']
    adapted=ROOT/'runs/phase3/local_source_full';adapted.mkdir(exist_ok=True)
    config.update(adapter_file=str((folder/'adapter.pt').relative_to(ROOT)),adapter_sha256=status['adapter_sha256'],
        adapter_config={k:training[k] for k in ('rank','alpha')},
        triage_override=report['triage_override'],
        expansion_evidence=dict(n=report['n'],delta=report['delta'],target_mix_sensitivity=report['target_mix_sensitivity']['delta']),
        decision='Complete 1307-question paired evaluation of the unchanged final adapter. Reuse identical 160-case outputs; additionally report the remaining 1147. No category routing or tuning.')
    cp=adapted/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Full evaluation config changed')
    atomic_json(cp,config);save_rows(adapted/'inputs.jsonl',rows)
    for target,old_name in [(baseline,'local_plain_validation'),(adapted,'local_source_separate')]:
        old=ROOT/'runs/phase3'/old_name;previous=read_rows(old/'inputs.jsonl')
        assert len(previous)==160 and all(by_id[r['qid']]==r for r in previous)
        outputs=[r for r in read_rows(old/'predictions.jsonl') if r['method']=='plain']
        assert len(outputs)==160 and {r['qid'] for r in outputs}=={r['qid'] for r in previous}
        if not (target/'predictions.jsonl').exists():save_rows(target/'predictions.jsonl',outputs)
        atomic_json(target/'reused.json',dict(source=str(old.relative_to(ROOT)),n=160,
            predictions_sha256=hashlib.sha256((old/'predictions.jsonl').read_bytes()).hexdigest()))
    print('prepared 1307 paired questions, reusing 160 outputs per condition; 1147 new per condition',flush=True)


def score(folder,separate=False,full=False):
    import numpy as np
    from experiment import scorer,summarize
    suffix='full' if full else 'separate' if separate else 'validation'
    source=ROOT/'runs/phase3'/('local_source_control_full' if full else 'local_plain_validation' if separate else 'local_source_control_validation')
    target=ROOT/'runs/phase3'/('local_source_'+suffix)
    data={s['sample_id']:s for s in json.loads((ROOT/'data/locomo10.json').read_text())};metric=scorer()
    inputs=read_rows(source/'inputs.jsonl');expected={r['qid'] for r in inputs};combined=[];by={}
    assert read_rows(target/'inputs.jsonl')==inputs
    for method,run in [('base',source),('adapted',target)]:
        rows=[r for r in read_rows(run/'predictions.jsonl') if r['method']=='plain']
        assert len(rows)==len(expected) and {r['qid'] for r in rows}==expected
        for r in rows:
            name,qi=r['qid'].rsplit(':',1);q=data[name]['qa'][int(qi)]
            gold={s for e in q.get('evidence',[]) for s in re.findall(r'D\d+:\d+',str(e))}
            r.update(method=method,category=q['category'],f1=metric(r['hypothesis'],q),
                     evidence_recall=len(gold&set(r['source_ids']))/len(gold) if gold else None)
        by[method]={r['qid']:r for r in rows};combined.extend(rows)
    for qid in expected:assert by['base'][qid]['source_ids']==by['adapted'][qid]['source_ids']
    delta=np.array([by['adapted'][qid]['f1']-by['base'][qid]['f1'] for qid in sorted(expected)])
    rng=np.random.default_rng(20261111)
    ci=np.quantile([rng.choice(delta,len(delta),replace=True).mean() for _ in range(10000)],[.025,.975])
    capped={m:sum(not r['stopped_with_eos'] for r in rows.values()) for m,rows in by.items()}
    result=dict(complete=True,n=len(expected),metrics=summarize(combined),delta=float(delta.mean()),
        bootstrap_ci=ci.tolist(),wins=int((delta>0).sum()),losses=int((delta<0).sum()),ties=int((delta==0).sum()),
        capped_outputs=capped,training=json.loads((folder/'training_config.json').read_text()),
        training_status=json.loads((folder/'training_status.json').read_text()),
        triage_override=json.loads((target/'config.json').read_text()).get('triage_override'),
        expansion_gate_passed=bool(delta.mean()>=.03 and all(n<=.05*len(expected) for n in capped.values())),
        scope='complete conversations 2-9; training/validation dialogues excluded; dataset reused in research' if full else
              'existing eight-conversation cohort, reused in research' if separate else 'one heldout dialogue, not full benchmark')
    if separate:
        from collections import Counter
        training=result['training'];excluded={training['train_dialogue'],training['validation_dialogue']}
        counts=Counter(q['category'] for name,s in data.items() if name not in excluded
                       for q in s['qa'] if q['category'] in (1,2,3,4))
        before=result['metrics']['base']['by_category'];after=result['metrics']['adapted']['by_category']
        effects={c:after[c]['f1']-before[c]['f1'] for c in counts}
        result['target_mix_sensitivity']=dict(target_n=sum(counts.values()),target_category_counts=dict(counts),
            observed_category_counts={c:before[c]['n'] for c in counts},category_deltas=effects,
            delta=sum(counts[c]/sum(counts.values())*effects[c] for c in counts),
            interpretation='Assumes within-category effects transfer to the complete target cohort. Sensitivity only, not measured full-benchmark F1. The original triage gate is unchanged.')
    if full:
        selected={r['qid'] for r in read_rows(ROOT/'runs/phase3/local_source_separate/inputs.jsonl')}
        remaining=expected-selected;assert len(expected)==1307 and len(remaining)==1147
        confirmation=[r for r in combined if r['qid'] in remaining]
        result['selection_cohort_excluded']=dict(n=1147,metrics=summarize(confirmation),
            delta=float(np.mean([by['adapted'][qid]['f1']-by['base'][qid]['f1'] for qid in sorted(remaining)])),
            scope='The 160-question expansion cohort excluded. These questions were not used to train or choose this adapter; dataset was used in earlier research.')
    atomic_json(target/'scored.json',combined);atomic_json(target/'summary.json',result)
    atomic_json(ROOT/'reports'/('source_reader_'+suffix+'.json'),result)
    print(json.dumps({k:result[k] for k in ('n','delta','bootstrap_ci','wins','losses','ties','capped_outputs','expansion_gate_passed')}))
    print({m:s['f1'] for m,s in result['metrics'].items()},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','prepare-evaluation','score','prepare-separate','score-separate','prepare-full','run-full','score-full'])
    args=p.parse_args();folder=ROOT/'runs/phase3/source_reader';folder.mkdir(exist_ok=True)
    if args.command=='prepare-full':prepare_full(folder)
    elif args.command=='run-full':
        import subprocess
        torch_python='/home/wqshen/anaconda3-new/envs/user-memory-study/bin/python'
        try:
            for name in ('local_source_control_full','local_source_full'):
                run=ROOT/'runs/phase3'/name
                print('Running',name,flush=True)
                with (run/'generation_stdout.log').open('a') as log:
                    subprocess.run([torch_python,'-u','src/local_cad.py','generate','--run',name],
                                   cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            score(folder,full=True)
            atomic_json(folder/'full_pipeline_status.json',dict(complete=True,scoring_environment='existing CPU .venv',new_paid_calls=0))
        except Exception as error:
            atomic_json(folder/'full_pipeline_status.json',dict(complete=False,error=str(error)))
            raise
    elif args.command=='score-full':score(folder,full=True)
    elif args.command=='prepare-separate':prepare_evaluation(folder,True)
    elif args.command=='score-separate':score(folder,True)
    else:{'prepare':prepare,'train':train,'prepare-evaluation':prepare_evaluation,'score':score}[args.command](folder)


if __name__=='__main__':main()
