"""One fixed, cache-only committee decision; no claim of posterior sampling."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from api import ROOT,atomic_json,digest
from experiment import scorer,summarize


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cohort',choices=['dev','validation'],default='dev')
    args=parser.parse_args();cohort=args.cohort
    if cohort=='validation':
        assert json.loads((ROOT/'reports/committee_consensus_dev.json').read_text())['expansion_gate_passed']
    sf=ROOT/'runs/phase3'/('sf_deliberate_dev' if cohort=='dev' else 'sf_deliberate_validation')
    local=ROOT/'runs/phase3'/('local_cad_greedy_dev' if cohort=='dev' else 'local_plain_validation')
    methods=['deliberate','v2','local_plain'];candidates={m:{} for m in methods}
    fingerprints={}
    for folder in (sf,local):
        path=folder/'predictions.jsonl';fingerprints[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            value=json.loads(line)
            method=value['method'] if folder==sf else 'local_plain' if value['method']=='plain' else None
            if method not in methods:continue
            # Project only generation fields. Existing scored-cache labels never
            # reach the decision rule; true references are loaded after selection.
            row={key:value[key] for key in ('qid','hypothesis','context_tokens','source_ids')}
            assert row['qid'] not in candidates[method];candidates[method][row['qid']]=row
    ids=sorted(candidates['deliberate']);assert len(ids)==(80 if cohort=='dev' else 160)
    assert all(set(values)==set(ids) for values in candidates.values())
    metric=scorer();selected=[]
    for qid in ids:
        rows=[candidates[m][qid] for m in methods]
        assert all(r['source_ids']==rows[0]['source_ids'] and r['context_tokens']==rows[0]['context_tokens'] for r in rows)
        utilities=[[metric(a['hypothesis'],dict(answer=b['hypothesis'],category=4)) for b in rows] for a in rows]
        scores=[sum(utilities[i][j] for j in range(len(rows)) if j!=i)/(len(rows)-1) for i in range(len(rows))]
        best=max(range(len(rows)),key=lambda i:scores[i])
        selected.append(dict(rows[best],selected_reader=methods[best],committee_scores=scores))
    run=ROOT/'runs/phase3'/('committee_consensus_'+cohort);run.mkdir(exist_ok=True)
    config=dict(candidate_order=methods,tie_break='first; strongest existing deliberate reader',
        utility='mean symmetric Porter token F1 against other two model answers, no gold categories',
        scope='heterogeneous finite committee, not iid posterior samples or exact Bayes optimality',
        source_files_sha256=fingerprints,selection_hash=digest(selected),n=len(ids),
        decision='Require >=3 paired F1 points above deliberate on development before one existing validation comparison; no weights or utility search.')
    atomic_json(run/'config.json',config);atomic_json(run/'selected_answer_only.json',selected)
    data={s['sample_id']:s for s in json.loads((ROOT/'data/locomo10.json').read_text())}
    by=dict(candidates,committee={r['qid']:r for r in selected});scored=[]
    for method,values in by.items():
        for qid,r in values.items():
            name,qi=qid.rsplit(':',1);q=data[name]['qa'][int(qi)]
            gold={mid for e in q.get('evidence',[]) for mid in re.findall(r'D\d+:\d+',str(e))}
            scored.append(dict(r,method=method,category=q['category'],f1=metric(r['hypothesis'],q),
                               evidence_recall=len(gold&set(r['source_ids']))/len(gold) if gold else None))
    scores={m:{r['qid']:r['f1'] for r in scored if r['method']==m} for m in by}
    delta=np.array([scores['committee'][qid]-scores['deliberate'][qid] for qid in ids])
    rng=np.random.default_rng(20261111)
    ci=np.quantile(rng.choice(delta,(5000,len(delta)),replace=True).mean(axis=1),[.025,.975])
    summary=dict(config=config,metrics=summarize(scored),delta_vs_strongest=float(delta.mean()),
        bootstrap_ci=ci.tolist(),wins=int((delta>0).sum()),losses=int((delta<0).sum()),ties=int((delta==0).sum()),
        selected_reader_counts=dict(Counter(r['selected_reader'] for r in selected)),
        oracle_pool_f1=float(np.mean([max(scores[m][qid] for m in methods) for qid in ids])),
        oracle_interpretation='Gold-only diagnostic ceiling; not an executable selector or achieved performance.',
        expansion_gate_passed=bool(delta.mean()>=.03),new_model_calls=0,
        compute_interpretation='This cached analysis adds zero calls. Deployment needs both DeepSeek modes plus one local answer; no cost-reduction claim.')
    atomic_json(run/'scored.json',scored);atomic_json(ROOT/'reports'/('committee_consensus_'+cohort+'.json'),summary)
    print({k:summary[k] for k in ('delta_vs_strongest','bootstrap_ci','wins','losses','ties','selected_reader_counts','oracle_pool_f1','expansion_gate_passed')})
    print({m:s['f1'] for m,s in summary['metrics'].items()})


if __name__=='__main__':main()
