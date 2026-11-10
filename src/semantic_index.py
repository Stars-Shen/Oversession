"""Question-blind semantic groups and a source-preserving diffusion index."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from api import ROOT,atomic_json,digest
from memory import BM25

PROMPT='''Organize these dated personal memory facts BEFORE any future questions are known.
Return JSON {"nodes":[{"entity":"person/entity", "topic":"specific semantic topic",
"summary":"precise connected pattern or event history with names and dates",
"kind":"pattern|event_cluster", "support":["F0"], "residual":["F1"]}]}.
Make about 35-50 focused nodes. Group semantically related facts across sessions. Include
recurring preferences, experiences, plans, relationships, turning points and their causes.
Connect common themes that use different words. Keep significant exceptions, changes, and
conflicting observations in residual, not averaged away. Distinguish actual from planned events.
Do not merge people or unrelated attributes. A pattern must have supporting facts from at least
two distinct sessions; mark it as tentative if it is an inference. Every support/residual entry
must be a supplied fact ID. Preserve exact named items and dates. Do not invent personal facts.
These summaries will serve as search keys; the original conversation remains authoritative.'''


def folder():return ROOT/'runs/phase3/semantic_index'


def build(name):
    from siliconflow import SiliconFlowAPI
    path=ROOT/'runs/memories_captions'/(name+'.json');data=json.loads(path.read_text())
    signature=digest(dict(memory=data,prompt=PROMPT));out=folder()/(name+'.json')
    if out.exists() and json.loads(out.read_text())['signature']==signature:return name,'cached'
    facts=data['facts'];alias={'F'+str(i):f for i,f in enumerate(facts)}
    content='\n'.join(json.dumps(dict(id=key,subject=f.get('subject',''),time=f.get('event_date'),
                                    observed=f['observed_at'],session=f['session'],text=f['text']),ensure_ascii=False)
                      for key,f in alias.items())
    api=SiliconFlowAPI();result=api.call(PROMPT,content,max_tokens=10000,json_output=True)
    nodes=[]
    for node in result['parsed'].get('nodes',[]):
        support=list(dict.fromkeys(x for x in node.get('support',[]) if x in alias))
        residual=list(dict.fromkeys(x for x in node.get('residual',[]) if x in alias))
        if not support:continue
        if node.get('kind')=='pattern' and len({alias[x]['session'] for x in support})<2:continue
        nodes.append(dict(entity=node['entity'],topic=node['topic'],summary=node['summary'],kind=node['kind'],
                          support=[alias[x]['id'] for x in support],residual=[alias[x]['id'] for x in residual]))
    atomic_json(out,dict(signature=signature,model=api.model,nodes=nodes,api_hash=result['request_hash'],usage=result['usage']))
    return name,len(nodes)


class SemanticIndex:
    def __init__(self,name,memory):
        data=json.loads((folder()/(name+'.json')).read_text());self.nodes=data['nodes'];self.signature=data['signature']
        self.vectors=np.load(folder()/(name+'.npz'))['nodes'].astype(np.float32)
        self.lexical=BM25([n['entity']+' '+n['topic']+' '+n['summary'] for n in self.nodes])
        nr=len(memory.turns);nn=len(self.nodes)
        # Sparse bipartite random walk. Both directions normalize by degree;
        # a broad abstraction redistributes its probability mass rather than
        # awarding a full independent vote to every supporting turn.
        self.edges=[];source_degree=np.zeros(nr);node_degree=np.zeros(nn)
        for ni,node in enumerate(self.nodes):
            refs={mid for fid in node['support']+node['residual'] for mid in memory.by_fact[fid]['sources']}
            for mid in sorted(refs,key=memory.turn_pos.get):
                ti=memory.turn_pos[mid];self.edges.append((ti,ni));source_degree[ti]+=1;node_degree[ni]+=1
        self.source_degree=np.maximum(source_degree,1);self.node_degree=np.maximum(node_degree,1)

    def retrieve(self,memory,question,budget,encoder,dense,vector):
        source=np.zeros(len(memory.turns));node=np.zeros(len(self.nodes))
        for ranking,is_fact in [(memory.raw_index.rank(question),False),(dense['raw'],False),
                                (memory.fact_index.rank(question),True),(dense['facts'],True)]:
            seen=set()
            for rank,(i,score) in enumerate(ranking):
                refs=memory.facts[i]['sources'] if is_fact else [memory.turns[i]['id']]
                for mid in refs:
                    if mid not in seen:source[memory.turn_pos[mid]]+=1/(40+rank);seen.add(mid)
        semantic=sorted(enumerate(self.vectors@vector),key=lambda item:(-item[1],item[0]))
        for ranking in (self.lexical.rank(question),semantic):
            for rank,(i,score) in enumerate(ranking[:10]):node[i]+=1/(40+rank)
        source_seed=.5*source/source.sum();node_seed=.5*node/node.sum()
        s=source_seed.copy();n=node_seed.copy()
        for _ in range(20):
            ns=np.zeros_like(s);nn=np.zeros_like(n)
            for ti,ni in self.edges:ns[ti]+=n[ni]/self.node_degree[ni];nn[ni]+=s[ti]/self.source_degree[ti]
            # Dangling sources restart at the same query-specific seed.
            dangling=float(s.sum()+n.sum()-ns.sum()-nn.sum())
            s=.5*source_seed+.5*(ns+dangling*source_seed)
            n=.5*node_seed+.5*(nn+dangling*node_seed)
        selected=[];sessions=set();used=0
        for ti in sorted(range(len(s)),key=lambda i:(-s[i],i)):
            t=memory.turns[ti];header='Session {} | {}\n'.format(t['session'],t['date']) if t['session'] not in sessions else ''
            line='[{}] {}: {}'.format(t['id'],t['speaker'],t['text']);cost=len(encoder.encode(header+line+'\n'))
            if used+cost<=budget:selected.append(ti);sessions.add(t['session']);used+=cost
        parts=[];sid=None
        for ti in sorted(selected):
            t=memory.turns[ti]
            if t['session']!=sid:parts.append('Session {} | {}'.format(t['session'],t['date']));sid=t['session']
            parts.append('[{}] {}: {}'.format(t['id'],t['speaker'],t['text']))
        text='\n'.join(parts)
        return dict(text=text,tokens=len(encoder.encode(text)),source_ids=[memory.turns[i]['id'] for i in selected],facts=[])


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['build','embed']);p.add_argument('--convs',default='0,1');args=p.parse_args()
    data=json.loads((ROOT/'data/locomo10.json').read_text());names=[data[int(i)]['sample_id'] for i in args.convs.split(',')]
    folder().mkdir(parents=True,exist_ok=True)
    if args.command=='build':
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(build,names):print(result,flush=True)
    else:
        from sentence_transformers import SentenceTransformer
        model=SentenceTransformer('/home/wqshen/models/Qwen3-Embedding-0.6B',device='cuda');model.max_seq_length=2048
        for name in names:
            nodes=json.loads((folder()/(name+'.json')).read_text())['nodes']
            vectors=model.encode([n['entity']+' '+n['topic']+' '+n['summary'] for n in nodes],normalize_embeddings=True,batch_size=32)
            np.savez_compressed(folder()/(name+'.npz'),nodes=vectors.astype(np.float16));print(name,len(nodes),flush=True)


if __name__=='__main__':main()
