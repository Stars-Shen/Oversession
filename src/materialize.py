"""Materialize completed cached extractions while the full extraction queue is running."""
import json
from api import ROOT,atomic_json,digest
from memory import EXTRACT,evolve
from longmem import sessions_for


def main():
    data=json.loads((ROOT/'data/longmemeval_s_cleaned.json').read_text())
    created=0
    for entry in data:
        path=ROOT/'runs/longmem_memories'/(entry['question_id']+'.json')
        if path.exists():continue
        sessions=sessions_for(entry)
        cache_paths={s['id']:ROOT/'runs/fact_cache'/(digest(dict(prompt=EXTRACT,session=s,model='gpt-6-luna'))+'.json')
                     for s in sessions}
        if not all(p.exists() for p in cache_paths.values()):continue
        extracted={sid:json.loads(p.read_text()) for sid,p in cache_paths.items()}
        facts=[f for s in sessions for f in extracted[s['id']]['facts']]
        atomic_json(path,dict(sessions=sessions,facts=evolve(facts),extraction=extracted,schema='vem-0.1'))
        created+=1
        print('materialized',entry['question_id'],len(facts),'facts',flush=True)
    print('Newly materialized:',created,flush=True)


if __name__=='__main__':main()
