"""Bounded evidence-first reading trial; question-only generation, separate scoring."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from api import ROOT,atomic_json,digest
from experiment import ANSWER,COMPACT,scorer,summarize
from evidence_contract import parse_answer

PROMPT='''Read the dated memories and bind evidence to the requested operation before answering.
Return one compact JSON object with these fields IN THIS ORDER:
{"observations":[{"s":["source ID"],"v":"brief relevant fact","t":"event date or unknown"}],
"operation":"lookup|compare|count|date|inference",
"check":"one short scope or calculation check", "answer":"concise answer"}.
Use at most three observations, grouping related list items if needed. Each observation must
cite IDs present in the memories and preserve exact relevant names, numbers and qualifications.
Distinguish the speaker from the person discussed. Bind the question's person, attribute and time.
For dates compare event times, not just mention times; use the source session date as the clock
for relative expressions. For counts distinguish repeated mentions from distinct actual events,
and plans from completed events. For comparisons retain each required side of the comparison.
When inference is requested, use supported personal preferences/experiences and reasonable common
sense; do not treat silence as proof of a negative. Include relevant counterevidence if present.
The check should be a short result, not a reasoning narrative. Keep the entire JSON around
140 tokens. If the evidence is insufficient, answer "No information available".
The following rules apply ONLY INSIDE the answer field:
'''+ANSWER+COMPACT


def prepare(folder,limit):
    base=ROOT/'runs/phase3/local_cad_greedy_dev'
    inputs=[json.loads(l) for l in (base/'inputs.jsonl').read_text().splitlines()]
    rows=[dict(r,system=PROMPT) for r in inputs[:limit]]
    config=json.loads((base/'config.json').read_text())
    config.update(methods=['evidence_first'],question_ids=[r['qid'] for r in rows],inputs_hash=digest(rows),
                  max_new_tokens=384,prompt_hash=digest(PROMPT),baseline_run='local_cad_greedy_dev',
                  intervention='same evidence; compact source-bound observations and operation before final answer',
                  pilot_decision='Expand from 24 to 80 only if paired F1 gain >=3 points, no truncation, and valid structured output >=95%.')
    cp=folder/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Run config changed')
    atomic_json(cp,config);(folder/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print('prepared',len(rows),'answer-free inputs',flush=True)


def score(folder):
    config=json.loads((folder/'config.json').read_text())
    generated=[json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()]
    raw=[r for r in generated if r['method']=='evidence_first']
    assert len(raw)==len(config['question_ids']) and {r['qid'] for r in raw}==set(config['question_ids'])
    base={r['qid']:r for r in json.loads((ROOT/'runs/phase3/local_cad_greedy_dev/scored.json').read_text()) if r['method']=='plain'}
    data=json.loads((ROOT/'data/locomo10.json').read_text());samples={s['sample_id']:s for s in data};metric=scorer()
    rows=[];errors=[];delta=[]
    for prediction in raw:
        before=base[prediction['qid']];after=dict(prediction)
        assert after['source_ids']==before['source_ids'] and after['context_tokens']==before['context_tokens']
        try:answer,trace=parse_answer(after['hypothesis'],set(after['source_ids']));after['evidence_state']=trace
        except (ValueError,KeyError,TypeError) as error:
            errors.append(dict(qid=after['qid'],error=str(error)));answer='No information available'
        after['raw_output']=after['hypothesis'];after['hypothesis']=answer
        name,qi=after['qid'].rsplit(':',1);q=samples[name]['qa'][int(qi)]
        after.update(category=q['category'],f1=metric(answer,q),evidence_recall=before['evidence_recall'])
        rows.extend([before,after]);delta.append(after['f1']-before['f1'])
    delta=np.array(delta);truncated=[dict(qid=r['qid'],method=r['method']) for r in generated if not r['stopped_with_eos']]
    result=dict(complete=True,n=len(raw),metrics=summarize(rows),delta=float(delta.mean()),
                wins=int((delta>0).sum()),losses=int((delta<0).sum()),ties=int((delta==0).sum()),
                format_or_source_errors=errors,truncated_outputs=truncated,
                expansion_gate_passed=bool(delta.mean()>=.03 and not truncated and len(errors)/len(raw)<=.05),
                config=config)
    atomic_json(folder/'scored.json',rows);atomic_json(folder/'summary.json',result)
    atomic_json(ROOT/'reports'/(folder.name+'.json'),result)
    print(json.dumps({k:result[k] for k in ('n','delta','wins','losses','ties','format_or_source_errors','truncated_outputs','expansion_gate_passed')},indent=2))
    print({m:s['f1'] for m,s in result['metrics'].items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','score'])
    p.add_argument('--run',default='local_evidence_first_pilot');p.add_argument('--limit',type=int,default=24)
    args=p.parse_args();folder=ROOT/'runs/phase3'/args.run;folder.mkdir(parents=True,exist_ok=True)
    prepare(folder,args.limit) if args.command=='prepare' else score(folder)


if __name__=='__main__':main()
