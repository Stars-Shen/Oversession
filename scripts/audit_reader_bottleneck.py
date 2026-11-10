"""One cached-output diagnostic; annotated-source coverage is not sufficiency."""
import json
from pathlib import Path
import re
import sys
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from api import ROOT,atomic_json


def main():
    samples={s['sample_id']:s for s in json.loads((ROOT/'data/locomo10.json').read_text())}
    result={}
    for run in ('local_cad_greedy_dev','local_plain_validation'):
        rows=[r for r in json.loads((ROOT/'runs/phase3'/run/'scored.json').read_text()) if r['method']=='plain']
        bins=defaultdict(list)
        for row in rows:
            name,qi=row['qid'].rsplit(':',1);q=samples[name]['qa'][int(qi)]
            gold={mid for e in q.get('evidence',[]) for mid in re.findall(r'D\d+:\d+',str(e))}
            key='all_annotated_sources_present' if gold<=set(row['source_ids']) else 'some_annotated_source_missing'
            bins[key].append(row['f1'])
        total_loss=sum(1-r['f1'] for r in rows)
        result[run]={key:dict(n=len(values),f1=sum(values)/len(values),
                     f1_loss_mass=sum(1-x for x in values),share_of_f1_loss=sum(1-x for x in values)/total_loss)
                     for key,values in bins.items()}
    result['limitations']='Annotations may omit needed context; complete annotated roots do not prove semantic sufficiency. F1 losses include formatting and reference mismatch, not only factual mistakes. This is descriptive, not a causal decomposition.'
    atomic_json(ROOT/'reports/reader_bottleneck_audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
