"""Pair reused original outputs with the changed calendar-scaffold outputs."""
import json
from pathlib import Path
import re
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from api import ROOT,atomic_json
from experiment import scorer,summarize


def main():
    base=ROOT/'runs/phase3/local_cad_greedy_dev';folder=ROOT/'runs/phase3/local_calendar_dev'
    original=[r for r in json.loads((base/'scored.json').read_text()) if r['method']=='plain']
    updates=[json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()]
    config=json.loads((folder/'config.json').read_text());changed={r['qid']:r for r in updates}
    assert len(changed)==len(updates) and set(changed)==set(config['question_ids'])
    data=json.loads((ROOT/'data/locomo10.json').read_text());samples={s['sample_id']:s for s in data};metric=scorer()
    combined=[];delta=[]
    for before in original:
        after=dict(changed.get(before['qid'],before),method='calendar')
        name,qi=before['qid'].rsplit(':',1);q=samples[name]['qa'][int(qi)]
        gold={mid for e in q.get('evidence',[]) for mid in re.findall(r'D\d+:\d+',str(e))}
        after.update(category=q['category'],f1=metric(after['hypothesis'],q),
                     evidence_recall=len(gold&set(after['source_ids']))/len(gold) if gold else None)
        assert after['context_tokens']<=3000
        combined.extend([before,after]);delta.append(after['f1']-before['f1'])
    delta=np.array(delta)
    result=dict(complete=True,n=len(original),new_generations=len(updates),reused_outputs=len(original)-len(updates),
                metrics=summarize(combined),delta=float(delta.mean()),wins=int((delta>0).sum()),
                losses=int((delta<0).sum()),ties=int((delta==0).sum()),
                truncated_outputs=[dict(qid=r['qid'],method=r['method']) for r in updates if not r['stopped_with_eos']],config=config)
    atomic_json(folder/'scored.json',combined);atomic_json(folder/'summary.json',result)
    atomic_json(ROOT/'reports/local_calendar_dev.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
