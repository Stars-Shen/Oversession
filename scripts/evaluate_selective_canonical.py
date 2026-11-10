"""Post-hoc development triage using the pre-existing question-only time rule."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from api import ROOT,atomic_json
from calendar_memory import TIME_QUERY


def main():
    folder=ROOT/'runs/phase3/local_canonical_values_dev'
    inputs={r['qid']:r for r in map(json.loads,(folder/'inputs.jsonl').read_text().splitlines())}
    rows=json.loads((folder/'scored.json').read_text())
    by={m:{r['qid']:r for r in rows if r['method']==m} for m in ('plain','canonical')}
    assert set(inputs)==set(by['plain'])==set(by['canonical'])
    selected=[qid for qid,row in inputs.items() if TIME_QUERY.search(row['question'])]
    delta=np.array([(by['canonical'][qid]['f1'] if qid in selected else by['plain'][qid]['f1'])
                    -by['plain'][qid]['f1'] for qid in inputs])
    baseline=np.mean([r['f1'] for r in by['plain'].values()])
    result=dict(n=len(delta),selected=len(selected),baseline_f1=float(baseline),
                selective_f1=float(baseline+delta.mean()),delta=float(delta.mean()),
                wins=int((delta>0).sum()),losses=int((delta<0).sum()),ties=int((delta==0).sum()),
                question_rule=TIME_QUERY.pattern,
                selection_scope='post-hoc representation selection on frozen development; question-only rule predates canonical trial; no pattern tuning',
                selected_qids=selected)
    atomic_json(ROOT/'reports/local_selective_canonical_exploratory.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
