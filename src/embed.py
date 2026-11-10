"""Offline local Qwen3 embeddings. Run with the existing torch environment."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
MODEL = '/home/wqshen/models/Qwen3-Embedding-0.6B'
QUERY = 'Instruct: Given a question about a conversation, retrieve relevant memories that answer it\nQuery: '


def encode_locomo(convs, memories, include_captions=False):
    data = json.loads((ROOT / 'data/locomo10.json').read_text())
    model = SentenceTransformer(MODEL, device='cuda:0', model_kwargs={'torch_dtype':torch.float16}, local_files_only=True)
    model.max_seq_length = 2048
    for ci in convs:
        sample = data[ci]
        memory = json.loads((memories / (sample['sample_id'] + '.json')).read_text())
        turns = [t for s in memory['sessions'] for t in s['turns']]
        docs = [t['speaker'] + ': ' + t['text'] for t in turns]
        facts = [f['text'] for f in memory['facts']]
        queries = [QUERY + q['question'] for q in sample['qa']]
        texts = docs + facts + queries
        fingerprint = hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()
        path = ROOT / ('runs/embeddings_captions' if include_captions else 'runs/embeddings') / (sample['sample_id'] + '.npz')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            cached = np.load(path)
            if str(cached['fingerprint']) == fingerprint:
                print('cached',sample['sample_id'],flush=True);continue
        vectors = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
        nd,nf = len(docs),len(facts)
        np.savez_compressed(path, raw=vectors[:nd],facts=vectors[nd:nd+nf],queries=vectors[nd+nf:],
                            fingerprint=fingerprint,model=MODEL,query_prompt=QUERY,
                            document_ids=np.array([t['id'] for t in turns]),fact_ids=np.array([f['id'] for f in memory['facts']]))
        print('embedded',sample['sample_id'],len(texts),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--convs',default='0,1')
    p.add_argument('--captions',action='store_true')
    args=p.parse_args()
    encode_locomo([int(x) for x in args.convs.split(',')],
                  ROOT/('runs/memories_captions' if args.captions else 'runs/memories'),args.captions)
