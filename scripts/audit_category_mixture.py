"""Category-mix sensitivity, never a substitute for missing full-run outputs."""
import json
from pathlib import Path
import sys
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from api import ROOT,atomic_json


def main():
    data=json.loads((ROOT/'data/locomo10.json').read_text())
    category={s['sample_id']+':'+str(i):str(q['category']) for s in data for i,q in enumerate(s['qa']) if q['category'] in (1,2,3,4)}
    full=Counter(category.values())
    development={s['sample_id']+':'+str(i) for s in data[:2] for i,q in enumerate(s['qa']) if q['category'] in (1,2,3,4)}
    phase=json.loads((ROOT/'reports/phase3.json').read_text())
    validation=set(phase['runs']['sf_deliberate_validation']['config']['question_ids'])
    cohorts={
        'validation':(phase['runs']['sf_deliberate_validation']['summary'],set(category)-development),
        'confirmation':(phase['phase3_development_and_validation_excluded']['summary'],set(category)-development-validation)}
    result=dict(full_category_counts=dict(full),cohorts={})
    for name,(summary,pool) in cohorts.items():
        counts=Counter(category[qid] for qid in pool)
        observed=summary['v2']['by_category'];after=summary['deliberate']['by_category']
        delta={c:after[c]['f1']-observed[c]['f1'] for c in observed}
        result['cohorts'][name]=dict(target_pool_n=len(pool),target_category_counts=dict(counts),
            observed_category_counts={c:v['n'] for c,v in observed.items()},
            observed_paired_delta=summary['deliberate']['f1']-summary['v2']['f1'],
            category_deltas=delta,
            target_mix_sensitivity_delta=sum(counts[c]/len(pool)*delta[c] for c in counts),
            full_mix_sensitivity_delta=sum(full[c]/sum(full.values())*delta[c] for c in full))
    result['interpretation']='Weighted deltas assume observed within-category means transfer to the target pool. Interrupted-run completion need not be random within categories. These are sensitivity calculations, not completed benchmark scores or replacements for missing answers; no new generations or rule tuning.'
    atomic_json(ROOT/'reports/category_mixture_audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
