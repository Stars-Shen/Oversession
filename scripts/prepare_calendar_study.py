"""Prepare only changed temporal inputs; unchanged answers remain cached."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tiktoken
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from api import ROOT,atomic_json,digest
from calendar_memory import TIME_QUERY,context
from memory import Memory


def main():
    base=ROOT/'runs/phase3/local_cad_greedy_dev';folder=ROOT/'runs/phase3/local_calendar_dev';folder.mkdir(parents=True,exist_ok=True)
    inputs=[json.loads(l) for l in (base/'inputs.jsonl').read_text().splitlines()]
    os.environ['TIKTOKEN_CACHE_DIR']=str(ROOT/'runs/tokenizer_cache')
    encoder=SimpleNamespace(encode=tiktoken.get_encoding('cl100k_base').encode_ordinary)
    memories={};changed=[]
    for row in inputs:
        if not TIME_QUERY.search(row['question']):continue
        name=row['qid'].rsplit(':',1)[0]
        if name not in memories:memories[name]=Memory(json.loads((ROOT/'runs/memories_captions'/(name+'.json')).read_text()))
        result=context(memories[name],row['source_ids'],encoder)
        if not result['anchors']:continue
        changed.append(dict(row,context=result['text'],context_tokens=result['tokens'],source_ids=result['source_ids'],calendar_anchors=result['anchors']))
    config=json.loads((base/'config.json').read_text())
    config.update(methods=['calendar'],question_ids=[r['qid'] for r in changed],inputs_hash=digest(changed),
                  baseline_run='local_cad_greedy_dev',original_question_ids=[r['qid'] for r in inputs],
                  intervention='question-only temporal gate; deterministic calendar offsets; same 3000-token final budget')
    cp=folder/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Run config changed')
    atomic_json(cp,config);(folder/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in changed))
    print('changed inputs',len(changed),'unchanged outputs reused',len(inputs)-len(changed))


if __name__=='__main__':main()
