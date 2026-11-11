"""One budgeted native 30B reader replacement, fixed 3K development evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from api import ROOT,atomic_json,digest
from full_history_reader import CONTROL,read_rows

FOLDER=ROOT/'runs/phase3/native_30b_dev'
MODEL=ROOT/'runs/phase3/models/Qwen3-30B-A3B-Thinking-2507'
REVISION='144afc2f379b542fdd4e85a1fcd5e1f79112d95d'


def prepare():
    rows=read_rows(CONTROL/'inputs.jsonl');old=json.loads((CONTROL/'config.json').read_text())
    metadata=json.loads((MODEL/'hf_metadata.json').read_text());assert metadata['sha']==REVISION
    config=dict(model=str(MODEL),revision=REVISION,question_ids=[r['qid'] for r in rows],
        inputs_hash=digest(rows),memory_sha256=old['memory_sha256'],context_budget=3000,
        model_config_sha256=hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest(),
        weights={f['rfilename']:f['lfs'] for f in metadata['siblings'] if f['rfilename'].endswith('.safetensors')},
        max_new_tokens=4096,batch_size=4,shards=2,seed=20261111,dtype='bfloat16',max_incomplete_fraction=.05,
        generation=dict(do_sample=True,temperature=.6,top_p=.95,top_k=20,min_p=0.),
        decision='One fixed reader policy. Expand only if >=3 F1 points above strongest cached DeepSeek deliberate and <=5% incomplete outputs. No temperature, prompt or reasoning-budget grid.',
        interpretation='Reader weights and native inference policy change together; not causal attribution to thinking alone or a new memory algorithm. Same 3K evidence and answer instructions.',
        split='reused 80 development questions; no adapter or training')
    cp=FOLDER/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    atomic_json(cp,config)
    (FOLDER/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print('prepared',len(rows),'answer-free inputs',flush=True)


def generate(shard):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    from local_cad import active_experts_forward
    from local_native import parse_completion
    config=json.loads((FOLDER/'config.json').read_text());rows=read_rows(FOLDER/'inputs.jsonl')
    assert digest(rows)==config['inputs_hash'] and shard in (0,1)
    assert hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest()==config['model_config_sha256']
    for name,metadata in config['weights'].items():assert (MODEL/name).stat().st_size==metadata['size']
    pf=FOLDER/('predictions.'+str(shard)+'.jsonl');saved=read_rows(pf) if pf.exists() else []
    done={r['qid'] for r in saved};assert len(done)==len(saved)
    batches=[(i,rows[start:start+4]) for i,start in enumerate(range(0,len(rows),4)) if i%2==shard]
    if all(r['qid'] in done for _,batch in batches for r in batch):print('shard already saved');return
    assert torch.cuda.device_count()==3
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side='left')
    closing=tok.convert_tokens_to_ids('</think>');assert tok.convert_ids_to_tokens(closing)=='</think>'
    placement={'model.embed_tokens':0,'model.rotary_emb':0,'model.norm':2,'lm_head':2}
    for i in range(48):placement['model.layers.'+str(i)]=i//16 if shard==0 else 0 if i<16 else 1 if i<39 else 2
    Qwen3MoeSparseMoeBlock.forward=active_experts_forward
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.bfloat16,
        device_map=placement,attn_implementation='sdpa').eval()
    device=model.get_input_embeddings().weight.device
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    runtime=dict(torch=torch.__version__,transformers=transformers.__version__,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),device_map={k:str(v) for k,v in placement.items()},
        batch_size=4,shard=shard,seed_policy='fixed seed per manifest batch; no independent per-question seed claim')
    atomic_json(FOLDER/('runtime.'+str(shard)+'.json'),runtime)
    with pf.open('a') as out:
        for batch_index,batch in batches:
            if all(r['qid'] in done for r in batch):continue
            assert not any(r['qid'] in done for r in batch),'partial batch requires explicit recovery'
            seed=(config['seed']+int(digest([r['qid'] for r in batch])[:8],16))%(2**31)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            texts=[tok.apply_chat_template([dict(role='system',content=r['system']),
                dict(role='user',content='Memories:\n'+r['context']+'\n\nQuestion: '+r['question'])],
                tokenize=False,add_generation_prompt=True) for r in batch]
            inputs=tok(texts,return_tensors='pt',padding=True,pad_to_multiple_of=8,add_special_tokens=False).to(device)
            start=time.time()
            with torch.inference_mode():
                output=model.generate(**inputs,**config['generation'],max_new_tokens=config['max_new_tokens'],
                    pad_token_id=tok.pad_token_id,logits_to_keep=1)
            records=[]
            for row,tokens,mask in zip(batch,output[:,inputs.input_ids.shape[1]:].tolist(),inputs.attention_mask):
                end=next((i+1 for i,t in enumerate(tokens) if t in eos),len(tokens));tokens=tokens[:end]
                records.append(dict(qid=row['qid'],method='native_thinking',
                    **parse_completion(tokens,closing,True,lambda part:tok.decode(part,skip_special_tokens=True)),
                    completion_tokens=len(tokens),input_tokens=int(mask.sum()),batch_seed=seed,batch_index=batch_index,
                    stopped_with_eos=bool(tokens and tokens[-1] in eos),batch_elapsed_seconds=time.time()-start,
                    context_tokens=row['context_tokens'],source_ids=row['source_ids']))
            out.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));out.flush();saved.extend(records)
            print('shard',shard,len(saved),'/ 40','tokens',sum(r['completion_tokens'] for r in records),
                'seconds',round(time.time()-start,2),'answers',[(r['qid'],r['hypothesis']) for r in records],flush=True)
    runtime['peak_allocated_gib']=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(3)]
    atomic_json(FOLDER/('runtime.'+str(shard)+'.json'),runtime)
    atomic_json(FOLDER/('status.'+str(shard)+'.json'),dict(complete=True,records=len(saved),expected_records=40))


def score():
    from full_history_reader import score as compare
    rows=[r for i in (0,1) for r in read_rows(FOLDER/('predictions.'+str(i)+'.jsonl'))]
    (FOLDER/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    compare(FOLDER,'native_thinking','Reader replacement with native reasoning and fixed sampling; one seed per batch. Same 3K evidence. Not a pure thinking toggle or new memory algorithm.')
    report=ROOT/'reports/native_30b_dev.json';result=json.loads(report.read_text())
    result['usage']={key:sum(r[key] for r in rows) for key in ('input_tokens','completion_tokens','reasoning_tokens')}
    atomic_json(report,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','generate','score']);p.add_argument('--shard',type=int,choices=[0,1]);args=p.parse_args()
    FOLDER.mkdir(parents=True,exist_ok=True)
    if args.command=='generate':generate(args.shard)
    else:globals()[args.command]()
