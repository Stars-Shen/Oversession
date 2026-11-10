"""Reuse valid first attempts; constrain one retry only when the contract fails."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from api import ROOT,atomic_json,digest
from evidence_first_study import PROMPT
from evidence_contract import parse_answer,source_schema


def main():
    base=ROOT/'runs/phase3/local_cad_greedy_dev';pilot=ROOT/'runs/phase3/local_evidence_first_pilot'
    folder=ROOT/'runs/phase3/local_evidence_first_guarded_dev';folder.mkdir(parents=True,exist_ok=True)
    inputs=[dict(json.loads(l),system=PROMPT) for l in (base/'inputs.jsonl').read_text().splitlines()]
    old_inputs={r['qid']:r for r in [json.loads(l) for l in (pilot/'inputs.jsonl').read_text().splitlines()]}
    first=[json.loads(l) for l in (pilot/'predictions.jsonl').read_text().splitlines()]
    by_qid={r['qid']:r for r in inputs};accepted=[]
    for row in first:
        assert old_inputs[row['qid']]==by_qid[row['qid']]
        try:parse_answer(row['hypothesis'],set(row['source_ids']))
        except (ValueError,KeyError,TypeError):continue
        accepted.append(dict(row,reused_first_attempt=True))
    inputs=[dict(r,schema=source_schema(r['source_ids'])) for r in inputs]
    config=json.loads((base/'config.json').read_text())
    config.update(methods=['evidence_first'],inputs_hash=digest(inputs),max_new_tokens=384,
                  prompt_hash=digest(PROMPT),baseline_run='local_cad_greedy_dev',guided_repair=True,
                  xgrammar_dependency_path='/home/wqshen/OverSession/.venv-hypermem-faithful/lib/python3.11/site-packages',
                  initial_attempts_file='runs/phase3/local_evidence_first_pilot/predictions.jsonl',
                  initial_attempts_hash=digest(first),reused_first_attempts=len(accepted),
                  intervention='unchanged evidence-first prompt; accept valid first attempt, otherwise one source-enum-constrained retry',
                  decision='No further interface/prompt grid. Expand only with >=3 F1 points and no truncation on 80 development questions.')
    cp=folder/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Run config changed')
    atomic_json(cp,config);(folder/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in inputs))
    pf=folder/'predictions.jsonl'
    if not pf.exists():pf.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted))
    print('scope',len(inputs),'valid first attempts reused',len(accepted),'pending',len(inputs)-len(accepted))


if __name__=='__main__':main()
