"""Fixed Luna medium reader through Flex, reusing all old low controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from api import ROOT,atomic_json,digest
from full_history_reader import CONTROL,read_rows
from openai_flex import FlexAPI,request_body,luna_cost

FOLDER=ROOT/'runs/phase3/openai_flex_reasoning_dev'


def prepare():
    rows=read_rows(CONTROL/'inputs.jsonl');original=json.loads((CONTROL/'config.json').read_text())
    low=[]
    for row in rows:
        body=request_body('gpt-6-luna',row['system'],'Memories:\n'+row['context']+'\n\nQuestion: '+row['question'],768,False,'low')
        old=json.loads((ROOT/'runs/api_cache'/(digest(body)+'.json')).read_text())
        low.append(dict(row,hypothesis=old['text'],method='luna_low',usage=old['usage'],request_hash=digest(body)))
    # UTF-8 byte length plus 1024 hidden/prompt overhead tokens is a conservative
    # planning proxy here, not a tokenizer guarantee or provider invoice limit.
    quote=sum((len((r['system']+'Memories:\n'+r['context']+'\n\nQuestion: '+r['question']).encode())+1024)*.0625+4096*.25 for r in rows)/1e6
    assert len(rows)==80 and quote<=.20
    config=dict(model='gpt-6-luna',reasoning='medium',max_new_tokens=4096,service_tier='flex',
        question_ids=[r['qid'] for r in rows],inputs_hash=digest(rows),memory_sha256=original['memory_sha256'],
        context_budget=3000,max_incomplete_fraction=.05,workers=4,
        conservative_planning_estimate_usd=quote,planning_estimate_limit_usd=.20,
        decision='One fixed policy; >=3 F1 points above strongest cached control and <=5% incomplete before expansion. No effort/model/budget grid.',
        cost_policy='Exact cache first; Flex only; no automatic retries, Standard fallback or output-limit doubling.',
        scope='Reused 80 development questions, frozen writer/context. All low controls reused; provider prices are estimates, not an invoice.')
    cp=FOLDER/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    atomic_json(cp,config);atomic_json(FOLDER/'baseline_low.json',low)
    (FOLDER/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print({'prepared':len(rows),'reused_low_controls':len(low),'conservative_planning_estimate_usd':quote},flush=True)


def generate():
    config=json.loads((FOLDER/'config.json').read_text());rows=read_rows(FOLDER/'inputs.jsonl')
    assert digest(rows)==config['inputs_hash']
    pilot=json.loads((ROOT/'runs/phase3/openai_flex_pilot.json').read_text())
    assert pilot.get('served_service_tier')=='flex' and pilot.get('complete'),'Flex pilot must succeed before expansion'
    pf=FOLDER/'predictions.jsonl';saved=read_rows(pf) if pf.exists() else [];done={r['qid'] for r in saved}
    assert len(done)==len(saved) and done<=set(config['question_ids'])
    pending=[r for r in rows if r['qid'] not in done];client=FlexAPI()
    def answer(row):
        result=client.call(row['system'],'Memories:\n'+row['context']+'\n\nQuestion: '+row['question'],
            max_tokens=config['max_new_tokens'],effort=config['reasoning'])
        return dict(qid=row['qid'],method='luna_medium',hypothesis=result['text'],complete=result.get('complete',True),
            source_ids=row['source_ids'],context_tokens=row['context_tokens'],usage=result['usage'],
            request_hash=result['request_hash'],new_paid_call=result['new_paid_call'],
            estimated_new_cost_usd=result['estimated_new_cost_usd'],served_service_tier=result.get('served_service_tier'),
            elapsed_seconds=result['elapsed_seconds'])
    atomic_json(FOLDER/'status.json',dict(complete=False,phase='generating',expected=80))
    try:
        with ThreadPoolExecutor(max_workers=4) as pool,pf.open('a') as out:
            for start in range(0,len(pending),4):
                for result in pool.map(answer,pending[start:start+4]):
                    out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush();saved.append(result)
                    print(len(saved),'/80',result['qid'],repr(result['hypothesis']),result['estimated_new_cost_usd'],flush=True)
                    if result['new_paid_call'] and result['served_service_tier']!='flex':raise RuntimeError('Unconfirmed Flex service; stop further batches')
        atomic_json(FOLDER/'status.json',dict(complete=True,records=len(saved),expected=80))
    except Exception as error:
        atomic_json(FOLDER/'status.json',dict(complete=False,records=len(saved),error=str(error)));raise


def score():
    import numpy as np
    from experiment import scorer,summarize
    from full_history_reader import score as compare
    compare(FOLDER,'luna_medium','Same Luna model and 3K evidence; medium/4096 versus cached low/768. Flex transport records actual served tier. Reused development; no memory-algorithm claim.')
    report=ROOT/'reports/openai_flex_reasoning_dev.json';summary=json.loads(report.read_text())
    scored=json.loads((FOLDER/'scored.json').read_text());metric=scorer()
    samples={s['sample_id']:s for s in json.loads((ROOT/'data/locomo10.json').read_text())}
    low=json.loads((FOLDER/'baseline_low.json').read_text());candidate={r['qid']:r for r in scored if r['method']=='luna_medium'}
    for row in low:
        name,qi=row['qid'].rsplit(':',1);q=samples[name]['qa'][int(qi)]
        row.update(category=q['category'],f1=metric(row['hypothesis'],q),evidence_recall=candidate[row['qid']]['evidence_recall'])
    delta=np.array([candidate[r['qid']]['f1']-r['f1'] for r in low])
    ci=np.quantile(np.random.default_rng(20261111).choice(delta,(5000,len(delta)),replace=True).mean(1),[.025,.975])
    scored+=low;summary['metrics']=summarize(scored)
    summary['comparisons']['luna_low']=dict(delta=float(delta.mean()),bootstrap_ci=ci.tolist(),wins=int((delta>0).sum()),losses=int((delta<0).sum()),ties=int((delta==0).sum()))
    pilot=json.loads((ROOT/'runs/phase3/openai_flex_pilot.json').read_text());records=read_rows(FOLDER/'predictions.jsonl')
    summary['new_api_cost_estimate_usd']=pilot['estimated_new_cost_usd']+sum(r['estimated_new_cost_usd'] for r in records)
    summary['actual_new_paid_calls']=int(pilot['new_paid_call'])+sum(r['new_paid_call'] for r in records)
    summary['new_paid_calls']=summary['actual_new_paid_calls']
    summary['usage']={k:sum(r['usage'][k] for r in records) for k in ('input_tokens','output_tokens','total_tokens')}
    summary['cost_comparison']=dict(same_medium_usage_at_standard_usd=sum(luna_cost(r['usage'],False) for r in records),
        old_low_completed_usage_at_standard_usd=sum(luna_cost(r['usage'],False) for r in low),old_low_reuse_new_cost_usd=0.,
        interpretation='Current public-rate estimates, not invoices; old Standard reprice is hypothetical. Reasoning tokens are included in output; pilot counted once.')
    summary['strongest_cached_control_f1']=max(summary['metrics'][m]['f1'] for m in ('luna_low','retrieved_plain','deepseek_deliberate'))
    summary['expansion_gate_passed']=summary['metrics']['luna_medium']['f1']-summary['strongest_cached_control_f1']>=.03 and len(summary['incomplete_outputs'])<=4
    atomic_json(FOLDER/'scored.json',scored);atomic_json(report,summary)
    print({'f1':{m:r['f1'] for m,r in summary['metrics'].items()},'estimated_usd':summary['new_api_cost_estimate_usd'],'new_calls':summary['actual_new_paid_calls'],'gate':summary['expansion_gate_passed']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','generate','score']);args=p.parse_args()
    FOLDER.mkdir(parents=True,exist_ok=True);globals()[args.command]()
