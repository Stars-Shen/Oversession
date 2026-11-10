"""Report paired phase-3 measurements from completed, provider-audited artifacts."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from api import ROOT,atomic_json
from experiment import summarize
from siliconflow import MODEL


def paired(rows,baseline,candidate):
    b={r['qid']:r for r in rows if r['method']==baseline}
    c={r['qid']:r for r in rows if r['method']==candidate}
    assert set(b)==set(c)
    delta=np.array([c[qid]['f1']-b[qid]['f1'] for qid in sorted(b)])
    samples=np.random.default_rng(20261111).choice(delta,(5000,len(delta)),replace=True).mean(axis=1)
    return dict(n=len(delta),mean=float(delta.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),
                wins=int((delta>0).sum()),losses=int((delta<0).sum()),ties=int((delta==0).sum()))


def read_run(name,allow_partial=False):
    folder=ROOT/'runs/phase3'/name;config=json.loads((folder/'config.json').read_text())
    rows=[json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines()]
    expected={(qid,m) for qid in config['question_ids'] for m in config['methods']}
    actual=[(r['qid'],r['method']) for r in rows]
    assert len(actual)==len(set(actual)) and set(actual)<=expected
    complete=set(actual)==expected
    assert complete or allow_partial
    for r in rows:
        assert r['context_tokens']<=config['budget']
        cached=json.loads((ROOT/'runs/phase3/api_cache'/(r['api_hash']+'.json')).read_text())
        assert cached['model']==MODEL and cached['provider']=='SiliconFlow' and cached['text']==r['hypothesis']
    common=set.intersection(*[{r['qid'] for r in rows if r['method']==m} for m in config['methods']])
    evaluated=rows if complete else [r for r in rows if r['qid'] in common]
    result=dict(config=config,complete=complete,expected_records=len(expected),stored_records=len(rows),
                paired_questions=len(common),summary=summarize(evaluated),
                paired={m:paired(evaluated,'v2',m) for m in config['methods'] if m!='v2'})
    for method in config['methods']:
        group=[r for r in evaluated if r['method']==method]
        result['summary'][method]['llm_tokens']=dict(
            prompt=sum(r['usage']['prompt_tokens'] for r in group),
            completion=sum(r['usage']['completion_tokens'] for r in group),
            reasoning=sum(r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) for r in group))
    return result,evaluated


def main():
    dev=['sf_baseline_dev','sf_coverage_dev','sf_table_dev','sf_graph_dev','sf_deliberate_dev','sf_cross_rank_dev']
    optional=['sf_cross_rank_validation','sf_deliberate_validation','sf_deliberate_full']
    report=dict(benchmark='LoCoMo categories 1-4, official caption protocol',answer_model=MODEL,
                provider='SiliconFlow',new_openai_calls=0,final_context_budget=3000,
                writer='frozen v2 Luna-written memories; this is not an end-to-end DeepSeek writer result',
                runs={},limitations=[
                    'Development hypotheses share the same 80 questions; development gains are selection evidence.',
                    'Conversations 2-9 are excluded from phase-3 development, but were used in the earlier project.',
                    'Paper judge accuracies are not comparable with this official F1 metric.',
                    'Compiler and native thinking controls use additional inference compute.',
                    'The nonthinking answer limit is 768 tokens; thinking uses 4096, with one doubling on truncation.',
                    'Cross ranking adds local GPU compute; final context and answer model are otherwise matched.'])
    report['dataset_sha256']=hashlib.sha256((ROOT/'data/locomo10.json').read_bytes()).hexdigest()
    report['official_scorer_sha256']=hashlib.sha256((ROOT/'third_party/locomo_evaluation.py').read_bytes()).hexdigest()
    for name in dev+optional:
        if not (ROOT/'runs/phase3'/name/'summary.json').exists():
            if name!='sf_deliberate_full' or not (ROOT/'runs/phase3'/name/'predictions.jsonl').exists():continue
        result,rows=read_run(name,allow_partial=name=='sf_deliberate_full');report['runs'][name]=result
        if name=='sf_deliberate_full':
            heldout=[r for r in rows if not r['qid'].startswith(('conv-26:','conv-30:'))]
            report['phase3_development_conversations_excluded']=dict(summary=summarize(heldout),paired=paired(heldout,'v2','deliberate'))
            validation=set(json.loads((ROOT/'runs/phase3/sf_deliberate_validation/config.json').read_text())['question_ids'])
            confirm=[r for r in heldout if r['qid'] not in validation]
            report['phase3_development_and_validation_excluded']=dict(summary=summarize(confirm),paired=paired(confirm,'v2','deliberate'))
            if not result['complete']:
                report['full_evaluation_status']='Incomplete: SiliconFlow HTTP 402 insufficient balance. Partial summaries use only completed paired questions and are not 1540-question results.'
            report['actual_reasoning_usage']=dict(
                nonthinking=sum(r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) for r in rows if r['method']=='v2'),
                thinking=sum(r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) for r in rows if r['method']=='deliberate'))
            report['previous_v2_report']='reports/final_v2.json; reader differs, do not attribute cross-model differences to a memory change'
    report['code_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (ROOT/'src').glob('*.py') if p.name in
                          ['siliconflow.py','phase3_experiment.py','semantic_index.py','cross_rank.py','evidence_table.py','coverage_memory.py']}
    atomic_json(ROOT/'reports/phase3.json',report)
    for name,run in report['runs'].items():
        print(name,'complete='+str(run['complete']),'paired_n='+str(run['paired_questions']),
              {m:round(s['f1']*100,3) for m,s in run['summary'].items()})


if __name__=='__main__':main()
