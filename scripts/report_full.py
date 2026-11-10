"""Generate the final comparison only from complete, current benchmark artifacts."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from api import ROOT,atomic_json


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def index(entries,method,expected):
    selected=[r for r in entries if r['method']==method]
    out={r['qid']:r for r in selected}
    if set(out)!=expected or len(selected)!=len(expected):
        raise RuntimeError('Incomplete or duplicate benchmark coverage: '+method+' '+str(len(selected)))
    return out


def compare(base,candidate,metric,ids):
    a=np.array([base[q][metric] for q in sorted(ids)],dtype=float)
    b=np.array([candidate[q][metric] for q in sorted(ids)],dtype=float)
    delta=b-a
    return dict(n=len(ids),baseline=float(a.mean()),candidate=float(b.mean()),delta_pp=float(delta.mean()*100),
                wins=int((delta>0).sum()),losses=int((delta<0).sum()),ties=int((delta==0).sum()))


def main():
    dataset_path=ROOT/'data/longmemeval_s_cleaned.json'
    dataset=json.loads(dataset_path.read_text())
    expected={e['question_id'] for e in dataset}
    assert len(dataset)==len(expected)==500
    dataset_hash=hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    predictions={};judgments={}
    for name,method in [('longmem_raw_full_v2','raw'),('longmem_hybrid_full_v2','hybrid'),('longmem_state_full_v2','consolidated')]:
        folder=ROOT/'runs'/name
        config=json.loads((folder/'config.json').read_text())
        assert set(config['question_ids'])==expected and config['dataset_sha256']==dataset_hash
        assert config['budget']==6000 and config['model']=='gpt-6-luna'
        predictions[method]=index(rows(folder/'predictions.jsonl'),method,expected)
        judgments[method]=index(rows(folder/'official_gpt4o.jsonl'),method,expected)
        for q,row in predictions[method].items():
            assert row['context_tokens']<=6000
            response=json.loads((ROOT/'runs/api_cache'/(row['answer_api_hash']+'.json')).read_text())
            assert response['model'].startswith('gpt-6-luna')
            label=judgments[method][q]
            assert label['judge_model']=='gpt-4o-2024-08-06'
            verdict=json.loads((ROOT/'runs/official_judge_cache'/(label['api_hash']+'.json')).read_text())
            assert verdict['model']=='gpt-4o-2024-08-06'
            assert label['correct']==('yes' in verdict['text'].lower())
    # Verify the evolving index retains the actual evidence ledgers used by the experiment.
    total_facts=0;total_tracks=0
    for q in expected:
        raw=json.loads((ROOT/'runs/longmem_memories'/(q+'.json')).read_text())
        evolved=json.loads((ROOT/'runs/longmem_consolidated'/(q+'.json')).read_text())
        assert raw['facts']==evolved['facts'] and raw['sessions']==evolved['sessions']
        source_ids={t['id'] for s in raw['sessions'] for t in s['turns']}
        assert all(set(f['sources'])<=source_ids for f in raw['facts'])
        assert all(set(t)<=set(['id','speaker','text','date','session','order']) for s in raw['sessions'] for t in s['turns'])
        total_facts+=len(raw['facts'])
        total_tracks+=sum(isinstance(t.get('entity'),str) and isinstance(t.get('attribute'),str) for t in evolved['tracks'])
    known=set(json.loads((ROOT/'runs/longmem_dev_ids.json').read_text()))
    known.update(json.loads((ROOT/'runs/longmem_validation_ids.json').read_text()))
    test=expected-known
    assert len(known)==72 and len(test)==428
    lme=dict(judge='gpt-4o-2024-08-06, official rubric and generation settings',
             full_vs_hybrid=compare(judgments['hybrid'],judgments['consolidated'],'correct',expected),
             unseen_428_vs_hybrid=compare(judgments['hybrid'],judgments['consolidated'],'correct',test),
             full_vs_bm25=compare(judgments['raw'],judgments['consolidated'],'correct',expected),
             per_type={kind:compare(judgments['hybrid'],judgments['consolidated'],'correct',
                                   {e['question_id'] for e in dataset if e['question_type']==kind})
                       for kind in sorted({e['question_type'] for e in dataset})},
             preserved_facts=total_facts,attribute_histories=total_tracks)
    locomo_data=json.loads((ROOT/'data/locomo10.json').read_text())
    ids={s['sample_id']+':'+str(i) for s in locomo_data for i,q in enumerate(s['qa']) if q['category']!=5}
    assert len(ids)==1540
    rs=rows(ROOT/'runs/locomo_full_packed_v2/predictions.jsonl')
    a=index(rs,'hybrid',ids);b=index(rs,'hybrid_packed',ids)
    assert all(r['context_tokens']<=3000 for r in rs)
    loco=dict(metric='official stemmed F1, nonadversarial categories 1-4, captions included',
              full=compare(a,b,'f1',ids),heldout_conversations=compare(a,b,'f1',{q for q in ids if q.split(':')[0] not in ('conv-26','conv-30')}),
              summary=json.loads((ROOT/'runs/locomo_full_packed_v2/summary.json').read_text()))
    report=dict(status='complete benchmark evidence',longmemeval=lme,locomo=loco,
                limitations=['LoCoMo conversations were evaluated across revisions.','The memory writer and final evidence compiler add inference cost; final reader evidence budgets are matched.','These same-backbone comparisons do not establish a global SOTA rank.'])
    atomic_json(ROOT/'reports/final_v2.json',report)
    print(json.dumps(dict(longmemeval=lme,locomo_full=loco['full']),indent=2))


if __name__=='__main__':main()
