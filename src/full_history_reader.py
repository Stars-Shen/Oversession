"""Fixed original-reader study: full raw history with causal prefix reuse."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from api import ROOT,atomic_json,digest
from prefix_memory import raw_history,prompt_ids,memory_prefix_ids

FOLDER=ROOT/'runs/phase3/full_history_dev'
CONTROL=ROOT/'runs/phase3/local_cad_greedy_dev'
STRONG=ROOT/'runs/phase3/sf_deliberate_dev'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prepare():
    import tiktoken
    os.environ['TIKTOKEN_CACHE_DIR']=str(ROOT/'runs/tokenizer_cache')
    encoder=tiktoken.get_encoding('cl100k_base')
    old=json.loads((CONTROL/'config.json').read_text());rows=[];histories={};hashes={}
    for row in read_rows(CONTROL/'inputs.jsonl'):
        name=row['qid'].rsplit(':',1)[0]
        if name not in histories:
            path=ROOT/'runs/memories_captions'/(name+'.json')
            histories[name]=raw_history(json.loads(path.read_text()))
            hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        text,ids=histories[name]
        rows.append(dict(row,context=text,source_ids=ids,context_tokens=len(encoder.encode_ordinary(text))))
    assert len(rows)==80 and hashes==old['memory_sha256']
    config={key:old[key] for key in ('model','max_new_tokens','dtype','model_config_sha256','tokenizer_config_sha256','seed')}
    config.update(question_ids=[r['qid'] for r in rows],inputs_hash=digest(rows),memory_sha256=hashes,
        original_inputs_hash=old['inputs_hash'],context_budget='all raw turns; no truncation',
        split='fixed 80 development questions from conversations 0 and 1',adapter=None,
        decoding='original greedy, max 192 new tokens, same answer instructions',
        cache='question-independent causal memory prefix; crop after each answer; no previous question or answer retained',
        device_layers=[16,23,9],provider='local; no paid calls',
        hypothesis='The 3K retrieval budget hides information needed by the original reader.',
        decision='Expand once only if paired F1 exceeds the existing strongest DeepSeek deliberate reader by >=3 points; no prompt, cache or layer-profile grid.')
    cp=FOLDER/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    atomic_json(cp,config)
    (FOLDER/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print('prepared',len(rows),'answer-free inputs; full histories',list(histories),flush=True)


def generate():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    from local_cad import active_experts_forward
    from prefix_memory import prefill,answer_from_prefix
    config=json.loads((FOLDER/'config.json').read_text());rows=read_rows(FOLDER/'inputs.jsonl')
    assert digest(rows)==config['inputs_hash']
    for name in ('model_config','tokenizer_config'):
        assert hashlib.sha256((Path(config['model'])/(name.replace('model_','')+'.json')).read_bytes()).hexdigest()==config[name+'_sha256']
    pf=FOLDER/'predictions.jsonl';saved=read_rows(pf) if pf.exists() else []
    done={r['qid'] for r in saved};assert len(done)==len(saved) and done<=set(config['question_ids'])
    pending=[r for r in rows if r['qid'] not in done]
    if not pending:print('all outputs already saved',flush=True);return
    tok=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    groups={}
    for row in pending:
        name=row['qid'].rsplit(':',1)[0];ids=prompt_ids(tok,row);prefix=memory_prefix_ids(tok,row)
        assert ids[:len(prefix)]==prefix
        if name not in groups:groups[name]=dict(prefix=prefix,rows=[])
        assert groups[name]['prefix']==prefix
        groups[name]['rows'].append((row,ids))
    assert torch.cuda.device_count()==3
    placement={'model.embed_tokens':0,'model.rotary_emb':0,'model.norm':2,'lm_head':2}
    for i in range(48):placement['model.layers.'+str(i)]=0 if i<16 else 1 if i<39 else 2
    Qwen3MoeSparseMoeBlock.forward=active_experts_forward
    started=time.time();print('loading original model; explicit layer placement 16/23/9',flush=True)
    model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,
        torch_dtype=torch.bfloat16,device_map=placement,attn_implementation='sdpa').eval()
    device=model.get_input_embeddings().weight.device
    gen=GenerationConfig(do_sample=False,max_new_tokens=config['max_new_tokens'],
        eos_token_id=model.generation_config.eos_token_id,pad_token_id=tok.pad_token_id)
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    runtime=dict(torch=torch.__version__,transformers=transformers.__version__,
        device_map={k:str(v) for k,v in model.hf_device_map.items()},
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),prefixes=[],
        compute_note='Prefill token accounting is local GPU work, not provider billing. KV reuse does not itself improve accuracy.')
    atomic_json(FOLDER/'runtime.json',runtime)
    with pf.open('a') as out:
        for name,group in groups.items():
            prefix=group['prefix'];start=time.time()
            print('prefilling',name,len(prefix),'tokens',flush=True)
            cache=prefill(model,torch.tensor([prefix],device=device))
            runtime['prefixes'].append(dict(conversation=name,tokens=len(prefix),elapsed_seconds=time.time()-start))
            atomic_json(FOLDER/'runtime.json',runtime)
            for row,tokens in group['rows']:
                start=time.time();ids=torch.tensor([tokens],device=device)
                output,cache=answer_from_prefix(model,ids,cache,len(prefix),gen)
                answer=output[0,len(tokens):].tolist()
                result=dict(qid=row['qid'],method='full_history',hypothesis=tok.decode(answer,skip_special_tokens=True).strip(),
                    context_tokens=row['context_tokens'],source_ids=row['source_ids'],completion_tokens=len(answer),
                    stopped_with_eos=bool(answer and answer[-1] in eos),prompt_tokens=len(tokens),
                    cached_prefix_tokens=len(prefix),question_prefill_tokens=len(tokens)-len(prefix),
                    elapsed_seconds=time.time()-start)
                out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush();saved.append(result)
                print(len(saved),'/ 80',row['qid'],repr(result['hypothesis']),round(result['elapsed_seconds'],2),flush=True)
            del cache
            torch.cuda.empty_cache()
    runtime.update(elapsed_seconds=time.time()-started,
        peak_allocated_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(3)])
    atomic_json(FOLDER/'runtime.json',runtime)
    atomic_json(FOLDER/'status.json',dict(complete=True,records=len(saved),expected_records=80))


def score(folder=FOLDER,method='full_history',interpretation=None):
    import numpy as np
    from experiment import scorer,summarize
    config=json.loads((folder/'config.json').read_text());metric=scorer()
    full=read_rows(folder/'predictions.jsonl');ids=config['question_ids']
    assert len(full)==80 and {r['qid'] for r in full}==set(ids)
    plain=[dict(r,method='retrieved_plain') for r in read_rows(CONTROL/'predictions.jsonl') if r['method']=='plain']
    strong=[dict(r,method='deepseek_deliberate') for r in read_rows(STRONG/'predictions.jsonl') if r['method']=='deliberate']
    assert all(len(rr)==80 and {r['qid'] for r in rr}==set(ids) for rr in (plain,strong))
    data={s['sample_id']:s for s in json.loads((ROOT/'data/locomo10.json').read_text())}
    scored=[]
    for r in full+plain+strong:
        name,qi=r['qid'].rsplit(':',1);q=data[name]['qa'][int(qi)]
        gold={mid for evidence in q.get('evidence',[]) for mid in re.findall(r'D\d+:\d+',str(evidence))}
        scored.append(dict(r,category=q['category'],f1=metric(r['hypothesis'],q),
            evidence_recall=len(gold&set(r['source_ids']))/len(gold) if gold else None))
    by={m:{r['qid']:r['f1'] for r in scored if r['method']==m}
        for m in (method,'retrieved_plain','deepseek_deliberate')}
    comparisons={}
    for control in ('retrieved_plain','deepseek_deliberate'):
        delta=np.array([by[method][qid]-by[control][qid] for qid in ids])
        ci=np.quantile(np.random.default_rng(20261111).choice(delta,(5000,len(delta)),replace=True).mean(1),[.025,.975])
        comparisons[control]=dict(delta=float(delta.mean()),bootstrap_ci=ci.tolist(),wins=int((delta>0).sum()),
            losses=int((delta<0).sum()),ties=int((delta==0).sum()))
    finished=lambda r:r.get('complete',r.get('stopped_with_eos',False))
    incomplete=[r['qid'] for r in full if not finished(r) or not r.get('closed_reasoning',True)]
    summary=dict(complete=True,n=80,config=config,metrics=summarize(scored),comparisons=comparisons,
        capped_outputs=[r['qid'] for r in full if not finished(r)],
        incomplete_outputs=incomplete,
        expansion_gate_passed=comparisons['deepseek_deliberate']['delta']>=.03 and len(incomplete)<=config.get('max_incomplete_fraction',1.)*len(full),
        interpretation=interpretation or 'Full history changes the information budget; cache reuse is compute optimization. Development is reused, not an untouched benchmark.',
        new_paid_calls=0)
    atomic_json(folder/'scored.json',scored);atomic_json(ROOT/'reports'/(folder.name+'.json'),summary)
    print(json.dumps({k:summary[k] for k in ('comparisons','capped_outputs','expansion_gate_passed')},indent=2))
    print({m:r['f1'] for m,r in summary['metrics'].items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','generate','score']);args=parser.parse_args()
    FOLDER.mkdir(parents=True,exist_ok=True)
    globals()[args.command]()
