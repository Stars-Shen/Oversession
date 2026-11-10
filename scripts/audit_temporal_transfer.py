"""Check prerequisites before transferring temporal geometry or change detection."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from api import ROOT,atomic_json


def main():
    ids=json.loads((ROOT/'runs/longmem_dev_ids.json').read_text());counts=Counter();hashes={}
    for qid in ids:
        path=ROOT/'runs/longmem_consolidated'/(qid+'.json')
        data=json.loads(path.read_text());facts={f['id']:f for f in data['facts']}
        hashes[qid]=hashlib.sha256(path.read_bytes()).hexdigest();counts['facts']+=len(facts)
        counts['single_state_facts']+=sum(f.get('single') and f['kind']=='state' for f in facts.values())
        for track in data['tracks']:
            if track['mode']!='state':continue
            observations=[facts[mid] for mid in track['members'] if facts[mid]['kind']=='state' and facts[mid].get('single')]
            counts['candidate_state_tracks']+=bool(observations)
            counts['multiple_observation_state_tracks']+=len(observations)>1
            counts['adjacent_pairs']+=max(0,len(observations)-1)
            counts['cross_session_adjacent_pairs']+=sum(a['session']!=b['session'] for a,b in zip(observations,observations[1:]))
    rng=np.random.default_rng(20261111);x=rng.normal(size=(512,2));y=rng.normal(size=(512,2));theta=rng.normal(size=512)
    def rotate(v):
        return np.stack([v[:,0]*np.cos(theta)-v[:,1]*np.sin(theta),
                         v[:,0]*np.sin(theta)+v[:,1]*np.cos(theta)],axis=-1)
    before=float(np.sum(x*y));after=float(np.sum(rotate(x)*rotate(y)))
    result=dict(scope='24 frozen LongMemEval development memories; no questions or answer labels read',
                question_ids=ids,memory_sha256=hashes,counts=dict(counts),
                shared_rotation=dict(before_dot=before,after_dot=after,absolute_difference=abs(before-after)),
                decisions=['Do not estimate a state changepoint likelihood from two candidate pairs.',
                           'Shared rotations alone preserve dot products; RoMem needs trained anisotropic relation weights.'])
    atomic_json(ROOT/'reports/temporal_transfer_audit.json',result)
    print(json.dumps(dict(counts=result['counts'],shared_rotation=result['shared_rotation']),indent=2))


if __name__=='__main__':main()
