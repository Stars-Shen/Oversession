"""Qwen3 raw-evidence embeddings for a strong LongMemEval retrieval baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT=Path(__file__).resolve().parents[1]
MODEL='/home/wqshen/models/Qwen3-Embedding-0.6B'
QUERY='Instruct: Given a question about a conversation, retrieve relevant memories that answer it\nQuery: '


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,default=3)
    p.add_argument('--device',required=True)
    args=p.parse_args()
    data=json.loads((ROOT/'data/longmemeval_s_cleaned.json').read_text())
    priority=set(json.loads((ROOT/'runs/longmem_dev_ids.json').read_text()))
    priority.update(json.loads((ROOT/'runs/longmem_validation_ids.json').read_text()))
    data=sorted(data,key=lambda e:(e['question_id'] not in priority,e['question_id']))[args.shard::args.shards]
    model=SentenceTransformer(MODEL,device=args.device,model_kwargs={'torch_dtype':torch.float16},local_files_only=True)
    model.max_seq_length=2048
    folder=ROOT/'runs/longmem_embeddings';folder.mkdir(exist_ok=True)
    for n,e in enumerate(data,1):
        texts=[];ids=[]
        for si,session in enumerate(e['haystack_sessions']):
            for ti,t in enumerate(session):
                texts.append(t['role']+': '+t['content'])
                ids.append('S'+str(si)+':T'+str(ti))
        texts.append(QUERY+e['question'])
        signature=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()
        path=folder/(e['question_id']+'.npz')
        if path.exists() and str(np.load(path)['fingerprint'])==signature:
            print(n,'/',len(data),e['question_id'],'cached',flush=True);continue
        vectors=model.encode(texts,batch_size=24,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        np.savez_compressed(path,raw=vectors[:-1],query=vectors[-1],document_ids=np.array(ids),
                            fingerprint=signature,model=MODEL,query_prompt=QUERY)
        print(n,'/',len(data),e['question_id'],len(ids),'embedded',flush=True)


if __name__=='__main__':main()
