"""Same source-priority retrieval, with frozen teacher facts as values."""
from collections import defaultdict


def retrieve(memory,question,budget,encoder,dense):
    # Obtain the unchanged four-view source order, without applying the final
    # rendering budget yet. The full text is never passed to a model.
    order=memory.retrieve_packed(question,1000000,encoder,dense)['source_ids']
    by_source=defaultdict(list)
    for fact in memory.facts:
        for mid in fact['sources']:by_source[mid].append(fact)
    packets=[];sessions=set();seen_facts=set();used=0
    roots=set();originals=[];fact_ids=[]
    for mid in order:
        turn=memory.by_turn[mid];all_facts=by_source[mid]
        facts=[f for f in all_facts if f['id'] not in seen_facts]
        if all_facts and not facts:continue
        if facts:
            block='\n'.join('[{}; canonical {}] {}'.format(','.join(f['sources']),f['kind'],f['text']) for f in facts)
            refs={source for f in facts for source in f['sources']}
        else:
            block='[{}] {}: {}'.format(mid,turn['speaker'],turn['text']);refs={mid}
        header='Session {} | {}\n'.format(turn['session'],turn['date']) if turn['session'] not in sessions else ''
        cost=len(encoder.encode(header+block+'\n'))
        if used+cost>budget:continue
        packets.append((memory.turn_pos[mid],turn['session'],turn['date'],block))
        sessions.add(turn['session']);used+=cost;roots.update(refs)
        if facts:
            fact_ids.extend(f['id'] for f in facts);seen_facts.update(f['id'] for f in facts)
        else:originals.append(mid)
    parts=[];session=None
    for pos,sid,date,block in sorted(packets):
        if sid!=session:parts.append('Session {} | {}'.format(sid,date));session=sid
        parts.append(block)
    text='\n'.join(parts);count=len(encoder.encode(text));assert count<=budget
    return dict(text=text,tokens=count,source_ids=sorted(roots,key=memory.turn_pos.get),
                original_source_ids=originals,fact_ids=fact_ids)
