"""Query-document relevance on short, complete dialogue windows.

Uses Qwen3-Reranker-0.6B's documented yes/no likelihood interface:
https://huggingface.co/Qwen/Qwen3-Reranker-0.6B (Apache-2.0 model).
No model training or benchmark answers enter scoring.
"""
import argparse
from collections import defaultdict
import json
import numpy as np
from api import ROOT,atomic_json,digest
from memory import Memory

INSTRUCTION='Given a question about a dated personal conversation, retrieve dialogue evidence that supports its answer. Match the correct person, relation and event; include relevant evidence for temporal, multi-event and inference questions.'


def candidates(memory,question,dense):
    scores=defaultdict(float)
    for ranking,is_fact in [(memory.raw_index.rank(question),False),(dense['raw'],False),
                            (memory.fact_index.rank(question),True),(dense['facts'],True)]:
        seen=set()
        for rank,(i,score) in enumerate(ranking):
            refs=memory.facts[i]['sources'] if is_fact else [memory.turns[i]['id']]
            for mid in refs:
                if mid not in seen:scores[mid]+=1/(40+rank);seen.add(mid)
    groups=[];seen=set()
    for mid in sorted(scores,key=lambda x:(-scores[x],memory.turn_pos[x]))[:120]:
        ti=memory.turn_pos[mid];session=memory.turns[ti]['session']
        refs=tuple(t['id'] for t in memory.turns[max(0,ti-1):ti+2] if t['session']==session)
        if refs not in seen:groups.append(refs);seen.add(refs)
    return groups


def retrieve_ranked(memory,ranking,budget,encoder):
    refs=set();sessions=set();used=0
    for item in sorted(ranking,key=lambda item:-item['score']):
        extra=[mid for mid in item['sources'] if mid not in refs]
        if not extra:continue
        t=memory.by_turn[extra[0]]
        header='Session {} | {}\n'.format(t['session'],t['date']) if t['session'] not in sessions else ''
        text=header+'\n'.join('[{}] {}: {}'.format(mid,memory.by_turn[mid]['speaker'],memory.by_turn[mid]['text']) for mid in extra)
        cost=len(encoder.encode(text+'\n'))
        if used+cost<=budget:refs.update(extra);sessions.add(t['session']);used+=cost
    parts=[];sid=None
    for mid in sorted(refs,key=memory.turn_pos.get):
        t=memory.by_turn[mid]
        if t['session']!=sid:parts.append('Session {} | {}'.format(t['session'],t['date']));sid=t['session']
        parts.append('[{}] {}: {}'.format(mid,t['speaker'],t['text']))
    text='\n'.join(parts)
    return dict(text=text,tokens=len(encoder.encode(text)),source_ids=sorted(refs,key=memory.turn_pos.get),facts=[])


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'runs/phase3/sf_baseline_dev/config.json'));args=p.parse_args()
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    model_path=ROOT/'runs/phase3/models/Qwen3-Reranker-0.6B'
    tok=AutoTokenizer.from_pretrained(model_path,padding_side='left')
    model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    prefix=tok.encode('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',add_special_tokens=False)
    suffix=tok.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False)
    yes=tok.convert_tokens_to_ids('yes');no=tok.convert_tokens_to_ids('no')
    data=json.loads((ROOT/'data/locomo10.json').read_text());by_name={s['sample_id']:i for i,s in enumerate(data)}
    ids=json.loads(open(args.config).read())['question_ids']
    chosen=[(by_name[qid.rsplit(':',1)[0]],int(qid.rsplit(':',1)[1])) for qid in ids]
    convs=sorted({ci for ci,qi in chosen});cache={}
    for ci in convs:
        name=data[ci]['sample_id'];mp=ROOT/'runs/memories_captions'/(name+'.json')
        cache[ci]=(Memory(json.loads(mp.read_text())),np.load(ROOT/'runs/embeddings_captions'/(name+'.npz')))
    out=ROOT/'runs/phase3/cross_rank';out.mkdir(parents=True,exist_ok=True)
    for ci,qi in chosen:
        question=data[ci]['qa'][qi]['question']
        name=data[ci]['sample_id'];path=out/(name+'_'+str(qi)+'.json')
        if path.exists():continue
        memory,emb=cache[ci];vector=emb['queries'][qi].astype(np.float32)
        dense={key:sorted(enumerate(emb[key].astype(np.float32)@vector),key=lambda item:(-item[1],item[0])) for key in ('raw','facts')}
        groups=candidates(memory,question,dense)
        documents=['\n'.join(memory.render_turn(memory.by_turn[mid]) for mid in refs) for refs in groups]
        scores=[]
        for start in range(0,len(groups),24):
            texts=['<Instruct>: '+INSTRUCTION+'\n<Query>: '+question+'\n<Document>: '+doc for doc in documents[start:start+24]]
            ids=tok(texts,add_special_tokens=False,truncation=True,max_length=1536-len(prefix)-len(suffix))['input_ids']
            batch=tok.pad({'input_ids':[prefix+item+suffix for item in ids]},padding=True,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                logits=model(**batch,logits_to_keep=1).logits[:,-1,[no,yes]].float()
                scores.extend(logits.softmax(dim=-1)[:,1].cpu().tolist())
        atomic_json(path,dict(model='Qwen/Qwen3-Reranker-0.6B',instruction=INSTRUCTION,question=question,
                              input_hash=digest(documents),max_length=1536,
                              ranking=[dict(sources=list(refs),score=score) for refs,score in zip(groups,scores)]))
        print(name,qi,len(groups),flush=True)


if __name__=='__main__':main()
