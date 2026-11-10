"""Grounded source-bundle coverage under a final-context budget.

A bundle contains all cited turns supporting a fact. Weighted set coverage is
submodular over bundles; shared source/header costs are not modular, so the
classic cardinality approximation bound is not claimed for this packing rule.
"""
import numpy as np


def retrieve_coverage(memory,question,budget,encoder,dense):
    raw_rank=memory.raw_index.rank(question)
    fact_rank=memory.fact_index.rank(question)
    raw_weight=np.zeros(len(memory.turns),dtype=float)
    fact_weight=np.zeros(len(memory.facts),dtype=float)
    # Limit reward to localized, question-relevant candidate views, without labels.
    for ranking,target in [(raw_rank,raw_weight),(dense['raw'],raw_weight),
                           (fact_rank,fact_weight),(dense['facts'],fact_weight)]:
        for rank,(i,score) in enumerate(ranking[:64]):target[i]+=1/(40+rank)
    if raw_weight.max()>0:raw_weight/=raw_weight.max()
    if len(fact_weight) and fact_weight.max()>0:fact_weight/=fact_weight.max()
    fact_ids=np.flatnonzero(fact_weight>0)
    refs={memory.turns[i]['id'] for i in np.flatnonzero(raw_weight>0)}
    for fi in fact_ids:refs.update(memory.facts[fi]['sources'])
    refs=sorted(refs,key=memory.turn_pos.get)
    if not refs:return dict(text='',source_ids=[],facts=[],tokens=0)
    pos={mid:i for i,mid in enumerate(refs)}
    source_sets={frozenset([mid]) for mid in refs}
    support={fi:frozenset(memory.facts[fi]['sources']) for fi in fact_ids}
    source_sets.update(support.values())
    bundles=sorted(source_sets,key=lambda b:tuple(sorted(memory.turn_pos[mid] for mid in b)))
    incidence=np.array([[mid in bundle for mid in refs] for bundle in bundles],dtype=bool)
    coverage=np.array([[support[fi]<=bundle for fi in fact_ids] for bundle in bundles],dtype=bool)
    features=np.concatenate([incidence,coverage],axis=1)
    weights=np.concatenate([np.array([raw_weight[memory.turn_pos[mid]] for mid in refs]),fact_weight[fact_ids]])
    lines={mid:'[{}] {}: {}'.format(mid,memory.by_turn[mid]['speaker'],memory.by_turn[mid]['text']) for mid in refs}
    costs=np.array([len(encoder.encode(lines[mid]+'\n')) for mid in refs])
    sessions=list(dict.fromkeys(memory.by_turn[mid]['session'] for mid in refs))
    session_pos={sid:i for i,sid in enumerate(sessions)}
    session_dates={t['session']:t['date'] for t in memory.turns}
    headers={sid:'Session {} | {}'.format(sid,session_dates[sid]) for sid in sessions}
    session_incidence=np.zeros((len(bundles),len(sessions)),dtype=bool)
    for bi,bundle in enumerate(bundles):
        for mid in bundle:session_incidence[bi,session_pos[memory.by_turn[mid]['session']]]=True
    header_cost=np.array([len(encoder.encode(headers[sid]+'\n')) for sid in sessions])
    selected_sources=np.zeros(len(refs),dtype=bool)
    selected_sessions=np.zeros(len(sessions),dtype=bool)
    covered=np.zeros(len(weights),dtype=bool)
    selected=np.zeros(len(bundles),dtype=bool)
    used=0
    while True:
        gains=features@(weights*(~covered))
        extra=incidence@(costs*(~selected_sources))+session_incidence@(header_cost*(~selected_sessions))
        eligible=(~selected)&(extra<=budget-used)&(gains>0)
        if not eligible.any():break
        ratio=np.where(eligible,gains/np.maximum(extra,1),-1)
        bi=int(np.argmax(ratio))
        used+=int(extra[bi]);selected[bi]=True
        selected_sources|=incidence[bi];selected_sessions|=session_incidence[bi]
        covered|=features[bi]
    source_ids=[mid for i,mid in enumerate(refs) if selected_sources[i]]
    parts=[];sid=None
    for mid in source_ids:
        current=memory.by_turn[mid]['session']
        if current!=sid:parts.append(headers[current]);sid=current
        parts.append(lines[mid])
    text='\n'.join(parts);count=len(encoder.encode(text))
    assert count<=budget
    return dict(text=text,source_ids=source_ids,facts=[],tokens=count,
                covered_fact_ids=[memory.facts[fi]['id'] for i,fi in enumerate(fact_ids) if covered[len(refs)+i]])
