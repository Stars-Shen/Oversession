"""Question-blind consolidation of immutable facts into evolving attribute histories."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from api import ROOT,API,atomic_json,digest

PROMPT='''Organize immutable conversation facts into entity-attribute histories. You do not know any
future questions. Return JSON {"tracks":[{"entity":"canonical entity", "attribute":"specific attribute or
recurring event family", "mode":"state|event_sequence|collection", "members":["F0","F7"]}]}.
Group co-referring attributes despite different wording, slot names, or scope labels. Examples:
team win/loss updates belong to one team-record track; camera lens purchases/ownership belong to
one lens-acquisition sequence; changes of employer belong to one current-employer state track.
Separate different people and distinct attributes. Never merge into vague topics like 'life' or
'equipment'. State means mutually exclusive values; collections and events retain multiple values.
Only output tracks with at least two distinct facts. A fact can belong to more than one specific
track when needed. Exclude irrelevant singleton facts. Do not rewrite facts, infer missing facts,
or declare which value is true. Preserve all observations, including apparent contradictions.'''


def consolidate(path,api):
    original=json.loads(path.read_text())
    facts=original['facts']
    aliases={'F'+str(i):f['id'] for i,f in enumerate(facts)}
    inputs=[dict(id='F'+str(i),subject=f['subject'],slot=f['slot'],scope=f.get('scope',''),
                 text=f['text'],observed_at=f['observed_at'],event_date=f.get('event_date'),kind=f['kind'])
            for i,f in enumerate(facts)]
    signature=digest(dict(prompt=PROMPT,facts=inputs,model=api.model))
    outpath=ROOT/'runs/longmem_consolidated'/path.name
    if outpath.exists() and json.loads(outpath.read_text()).get('consolidation_hash')==signature:return path.stem,'cached'
    result=api.call(PROMPT,json.dumps(inputs,ensure_ascii=False),max_tokens=12000,json_output=True)
    by_id={f['id']:f for f in facts}
    tracks=[]
    for t in result['parsed']['tracks']:
        if not isinstance(t.get('entity'),str) or not isinstance(t.get('attribute'),str):continue
        members=list(dict.fromkeys(aliases[x] for x in t['members'] if x in aliases))
        if len(members)<2:continue
        mode=t.get('mode','collection')
        if mode not in ('state','event_sequence','collection'):continue
        t=dict(t,mode=mode,members=sorted(members,key=lambda mid:(by_id[mid]['order'],mid)))
        tracks.append(t)
    atomic_json(outpath,dict(original,tracks=tracks,consolidation_hash=signature,consolidation_api_hash=result['request_hash']))
    return path.stem,len(tracks)


if __name__=='__main__':
    api=API()
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs=[pool.submit(consolidate,p,api) for p in sorted((ROOT/'runs/longmem_memories').glob('*.json'))]
        for job in as_completed(jobs):print('consolidated',*job.result(),flush=True)
