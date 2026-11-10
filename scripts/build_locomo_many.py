"""Build independent conversation memories with one bounded extraction queue."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from api import ROOT, API,atomic_json,digest
from memory import locomo_sessions,extract_session,evolve,EXTRACT

data=json.loads((ROOT/'data/locomo10.json').read_text())
convs=[int(x) for x in sys.argv[1].split(',')]
sessions={ci:locomo_sessions(data[ci]) for ci in convs}
results={ci:{} for ci in convs}
api=API()
with ThreadPoolExecutor(max_workers=24) as pool:
    jobs={pool.submit(extract_session,s,api):(ci,s['id']) for ci,ss in sessions.items() for s in ss}
    for count,job in enumerate(as_completed(jobs),1):
        ci,sid=jobs[job]
        results[ci][sid]=job.result()
        print(count,'/',len(jobs),data[ci]['sample_id'],sid,flush=True)
        if len(results[ci])==len(sessions[ci]):
            facts=[f for s in sessions[ci] for f in results[ci][s['id']]['facts']]
            memory=dict(sessions=sessions[ci],facts=evolve(facts),extraction=results[ci],schema='vem-0.1',
                        sample_id=data[ci]['sample_id'],extraction_prompt_hash=digest(EXTRACT))
            atomic_json(ROOT/'runs/memories'/(data[ci]['sample_id']+'.json'),memory)
            print('built',ci,len(facts),flush=True)
