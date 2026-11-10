"""Local Qwen study of context-aware decoding; no paid API calls.

CAD: Shi et al., NAACL 2024, https://aclanthology.org/2024.naacl-short.69/.
Transformers' unbatched classifier-free guidance implements the same logit rule
with guidance_scale=1+alpha. This reuses the installed library, not paper code.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from api import ROOT,atomic_json,digest

MODEL_PATH=Path('/home/wqshen/models/Qwen3-30B-A3B-Instruct-2507')


def active_experts_forward(self,hidden_states):
    """Same routing order/math; omit experts receiving zero tokens.

    Adapted from Transformers 4.51.3 Qwen3MoeSparseMoeBlock (Apache-2.0).
    The installed implementation invokes every expert, including empty tensors.
    """
    import torch
    batch,length,width=hidden_states.shape;flat=hidden_states.view(-1,width)
    router=self.gate(flat)
    weight,selected=torch.topk(router.softmax(dim=1,dtype=torch.float),self.top_k,dim=-1)
    if self.norm_topk_prob:weight/=weight.sum(dim=-1,keepdim=True)
    weight=weight.to(flat.dtype);result=torch.zeros_like(flat)
    mask=torch.nn.functional.one_hot(selected,num_classes=self.num_experts).permute(2,1,0)
    for expert in selected.unique(sorted=True).tolist():
        slot,token=torch.where(mask[expert])
        values=self.experts[expert](flat[None,token].reshape(-1,width))*weight[token,slot,None]
        result.index_add_(0,token,values.to(flat.dtype))
    return result.reshape(batch,length,width),router


def prepare(args,folder):
    from types import SimpleNamespace
    import numpy as np
    import tiktoken
    from experiment import ANSWER,COMPACT,choose_questions
    from memory import Memory
    os.environ['TIKTOKEN_CACHE_DIR']=str(ROOT/'runs/tokenizer_cache')
    encoder=SimpleNamespace(encode=tiktoken.get_encoding('cl100k_base').encode_ordinary)
    data=json.loads((ROOT/'data/locomo10.json').read_text())
    chosen=choose_questions(data,[int(x) for x in args.convs.split(',')],args.limit,20261111)
    cache={};rows=[];hashes={}
    for ci,qi,q in chosen:
        name=data[ci]['sample_id']
        if ci not in cache:
            p=ROOT/'runs/memories_captions'/(name+'.json');hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
            cache[ci]=(Memory(json.loads(p.read_text())),np.load(ROOT/'runs/embeddings_captions'/(name+'.npz')))
        memory,emb=cache[ci];vector=emb['queries'][qi].astype(np.float32)
        dense={key:sorted(enumerate(emb[key].astype(np.float32)@vector),key=lambda item:(-item[1],item[0])) for key in ('raw','facts')}
        ctx=memory.retrieve(q['question'],'hybrid_packed',3000,encoder,dense)
        assert ctx['tokens']<=3000
        rows.append(dict(qid=name+':'+str(qi),question=q['question'],system=ANSWER+COMPACT,
                         context=ctx['text'],source_ids=ctx['source_ids'],context_tokens=ctx['tokens']))
    config=dict(model=str(MODEL_PATH),alpha=.5,max_new_tokens=192,dtype='bfloat16',
                methods=['plain'] if args.plain_only else ['plain','cad'],question_ids=[r['qid'] for r in rows],seed=20261111,
                inputs_hash=digest(rows),memory_sha256=hashes,context_budget=3000,
                model_config_sha256=hashlib.sha256((MODEL_PATH/'config.json').read_bytes()).hexdigest(),
                tokenizer_config_sha256=hashlib.sha256((MODEL_PATH/'tokenizer_config.json').read_bytes()).hexdigest(),
                provider='local; no network',writer='frozen v2 Luna',split='development' if set(args.convs.split(','))<={'0','1'} else 'evaluation')
    config['decoding']='greedy; explicit do_sample=False override'
    config['routing_execution']='original expert order, empty experts skipped'
    cp=folder/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Run config changed')
    atomic_json(cp,config)
    (folder/'inputs.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    print('prepared',len(rows),'question-only generation inputs',flush=True)


def generate(args,folder):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    config=json.loads((folder/'config.json').read_text())
    rows=[json.loads(l) for l in (folder/'inputs.jsonl').read_text().splitlines()]
    assert digest(rows)==config['inputs_hash']
    pf=folder/'predictions.jsonl'
    records=[json.loads(l) for l in pf.read_text().splitlines()] if pf.exists() else []
    done={(r['qid'],r['method']) for r in records}
    pending=[(r,m) for r in rows for m in config['methods'] if (r['qid'],m) not in done]
    if not pending:print('all outputs already saved',flush=True);return
    tok=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    first_attempts={};compiler=None
    if config.get('guided_repair'):
        from evidence_contract import parse_answer
        sys.path.append(config['xgrammar_dependency_path'])
        import xgrammar as xgr
        from xgrammar.contrib.hf import LogitsProcessor
        vocab_size=json.loads((Path(config['model'])/'config.json').read_text())['vocab_size']
        compiler=xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tok,vocab_size=vocab_size))
        first=[json.loads(l) for l in (ROOT/config['initial_attempts_file']).read_text().splitlines()]
        assert digest(first)==config['initial_attempts_hash'];first_attempts={r['qid']:r for r in first}
    print('loading local model on visible GPUs',torch.cuda.device_count(),flush=True)
    Qwen3MoeSparseMoeBlock.forward=active_experts_forward
    model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,
            torch_dtype=torch.bfloat16,device_map='balanced',
            max_memory={i:'22GiB' for i in range(torch.cuda.device_count())},attn_implementation='sdpa').eval()
    assert all(str(device) not in ('cpu','disk') for device in model.hf_device_map.values())
    runtime=dict(torch=torch.__version__,transformers=transformers.__version__,
                 device_map={k:str(v) for k,v in model.hf_device_map.items()},
                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if 'cad' in config['methods']:runtime.update(alpha=config['alpha'],guidance_scale=1+config['alpha'])
    if compiler is not None:runtime['source_repair']='one xgrammar constrained retry on contract failure'
    atomic_json(folder/'runtime.json',runtime)
    device=model.get_input_embeddings().weight.device
    gen=GenerationConfig(do_sample=False,max_new_tokens=config['max_new_tokens'],
                         eos_token_id=model.generation_config.eos_token_id,pad_token_id=tok.pad_token_id)
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    with pf.open('a') as out:
        for row,method in pending:
            start=time.time()
            def prompt(content):
                return tok.apply_chat_template([dict(role='system',content=row['system']),dict(role='user',content=content)],
                                               tokenize=True,add_generation_prompt=True,return_tensors='pt').to(device)
            ids=prompt('Memories:\n'+row['context']+'\n\nQuestion: '+row['question'])
            extra={}
            if method=='cad':
                negative=prompt('Question: '+row['question'])
                extra=dict(guidance_scale=1+config['alpha'],negative_prompt_ids=negative,
                           negative_prompt_attention_mask=torch.ones_like(negative))
            def decode(options):
                with torch.inference_mode():
                    output=model.generate(ids,attention_mask=torch.ones_like(ids),generation_config=gen,
                                          do_sample=False,temperature=1.0,top_p=1.0,top_k=50,
                                          logits_to_keep=1,**options)
                answer_ids=output[0,ids.shape[1]:].tolist()
                return dict(qid=row['qid'],method=method,hypothesis=tok.decode(answer_ids,skip_special_tokens=True).strip(),
                            elapsed_seconds=time.time()-start,completion_tokens=len(answer_ids),
                            stopped_with_eos=bool(answer_ids and answer_ids[-1] in eos),
                            context_tokens=row['context_tokens'],source_ids=row['source_ids'])
            result=dict(first_attempts[row['qid']]) if row['qid'] in first_attempts else decode(extra)
            if compiler is not None:
                try:parse_answer(result['hypothesis'],set(row['source_ids']))
                except (ValueError,KeyError,TypeError) as error:
                    initial=result
                    constrained=dict(extra,logits_processor=[LogitsProcessor(compiler.compile_json_schema(row['schema']))])
                    result=decode(constrained);result.update(repair_used=True,initial_error=str(error),first_attempt=initial,
                                                            total_completion_tokens=initial['completion_tokens']+result['completion_tokens'])
            out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush();records.append(result)
            print(len(records),'/',len(rows)*len(config['methods']),row['qid'],method,
                  repr(result['hypothesis']),round(result['elapsed_seconds'],2),flush=True)
    atomic_json(folder/'status.json',dict(complete=True,records=len(records),expected_records=len(rows)*len(config['methods'])))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','generate','score'])
    p.add_argument('--convs',default='0,1');p.add_argument('--limit',type=int,default=80)
    p.add_argument('--run',default='local_cad_greedy_dev');p.add_argument('--plain-only',action='store_true');args=p.parse_args()
    folder=ROOT/'runs/phase3'/args.run;folder.mkdir(parents=True,exist_ok=True)
    if args.command=='prepare':prepare(args,folder)
    elif args.command=='generate':generate(args,folder)
    else:
        from experiment import scorer,summarize
        import numpy as np
        import re
        data=json.loads((ROOT/'data/locomo10.json').read_text());by_name={s['sample_id']:s for s in data};metric=scorer()
        rows=[json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()]
        config=json.loads((folder/'config.json').read_text())
        expected={(qid,m) for qid in config['question_ids'] for m in config['methods']}
        assert {(r['qid'],r['method']) for r in rows}==expected and len(rows)==len(expected)
        for row in rows:
            name,qi=row['qid'].rsplit(':',1);q=by_name[name]['qa'][int(qi)]
            gold={mid for evidence in q.get('evidence',[]) for mid in re.findall(r'D\d+:\d+',str(evidence))}
            row.update(category=q['category'],f1=metric(row['hypothesis'],q),
                       evidence_recall=len(gold&set(row['source_ids']))/len(gold) if gold else None)
        # Both modes use exactly the same retrieved evidence, already audited at preparation.
        result=dict(complete=True,metrics=summarize(rows),
                    truncated_outputs=[dict(qid=r['qid'],method=r['method']) for r in rows if not r['stopped_with_eos']])
        if {'plain','cad'}<=set(config['methods']):
            by_method={m:{r['qid']:r for r in rows if r['method']==m} for m in ('plain','cad')}
            delta=np.array([by_method['cad'][qid]['f1']-by_method['plain'][qid]['f1'] for qid in sorted(by_method['plain'])])
            result.update(delta=float(delta.mean()),wins=int((delta>0).sum()),losses=int((delta<0).sum()),ties=int((delta==0).sum()))
        atomic_json(folder/'scored.json',rows);atomic_json(folder/'summary.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
