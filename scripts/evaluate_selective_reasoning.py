"""Post-hoc evaluation of a question-only reasoning policy using saved outputs.

This does not make API calls or provide an independent confirmation result.
"""
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from api import ROOT,atomic_json,digest
from experiment import summarize

RULE=r'\b(?:when|how\s+(?:long|many|much|old)|before|after|since|between|earlier|later|first|last|latest|currently|now|date|year|month|week|day|likely|would|might|could|both|in common|considered|personality)\b'


def main():
    folder=ROOT/'runs/phase3/sf_deliberate_full'
    rows=[json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()]
    plain={r['qid']:r for r in rows if r['method']=='v2'}
    thinking={r['qid']:r for r in rows if r['method']=='deliberate'}
    ids=sorted(plain.keys()&thinking.keys());selected=[]
    for qid in ids:
        assert plain[qid]['source_ids']==thinking[qid]['source_ids']
        use_thinking=bool(re.search(RULE,plain[qid]['question'],flags=re.I))
        source=thinking[qid] if use_thinking else plain[qid]
        selected.append(dict(source,method='selective',selected_method=source['method']))
    result=dict(status='exploratory post-hoc analysis, not independent test evidence',
                reason='Question-operator policy motivated after inspecting existing error patterns; freeze before prospective testing.',
                rule=RULE,rule_hash=digest(RULE),input_run='sf_deliberate_full',
                input_scope='completed paired portion of the interrupted run, not the full 1540 questions',n=len(ids),
                selected_thinking=sum(r['selected_method']=='deliberate' for r in selected),
                metrics=summarize([plain[qid] for qid in ids]+[thinking[qid] for qid in ids]+selected))
    reasoning=lambda group:sum(r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) for r in group)
    actual_thinking=reasoning([thinking[qid] for qid in ids]);chosen_thinking=reasoning(selected)
    total=lambda group:sum(r['usage']['prompt_tokens']+r['usage']['completion_tokens'] for r in group)
    actual_total=total([thinking[qid] for qid in ids]);chosen_total=total(selected)
    result['counterfactual_deployment_usage']=dict(always_thinking_reasoning_tokens=actual_thinking,
            selective_reasoning_tokens=chosen_thinking,reasoning_token_reduction=1-chosen_thinking/actual_thinking,
            always_thinking_total_tokens=actual_total,selective_total_tokens=chosen_total,total_token_reduction=1-chosen_total/actual_total,
            note='Estimated deployment usage from observed branch usage. Both branches were actually generated in the research run; this is not an invoice saving.')
    atomic_json(ROOT/'reports/selective_reasoning_exploratory.json',result)
    print(json.dumps(dict(n=len(ids),f1={m:100*s['f1'] for m,s in result['metrics'].items()},
                          selected_thinking=result['selected_thinking'],usage=result['counterfactual_deployment_usage']),indent=2))


if __name__=='__main__':main()
