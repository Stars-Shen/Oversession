"""Request-local relational evidence; immutable source turns remain authoritative."""
import json

PROMPT = '''Build a small evidence table for the supplied question and dated conversation.
Return JSON, without an answer or a reasoning narrative:
{"rows":[{"subject":"person/entity", "predicate":"specific relation or event",
 "value":"precise relevant detail", "event_time":"event date/range or unknown",
 "mention_time":"date of source session", "episode_key":"same key for repeated mentions of one event",
 "status":"actual|planned|hypothetical|uncertain", "sources":["turn ID"]}],
 "inferences":[{"claim":"plausible implication with its uncertainty", "sources":["turn ID"]}],
 "constraints":[{"claim":"qualification, exception, or missing prerequisite", "sources":["turn ID"]}]}.
Select evidence that actually bears on the exact question; preserve all distinct relevant items.
Resolve pronouns using dialogue context. Separate the person speaking from the person discussed.
Separate event time from mention time; resolve relative dates against the source session date.
Do not replace an old event with a newer similar event. Repeated mentions are not new events.
For counts preserve one row per distinct actual event and keep planned events separate.
For temporal order retain anchor events and dates. For shared attributes retain both people's evidence.
For quotes retain the original wording. Keep exact names, numbers, places, and qualifications.
Include personal implications only when the question requests a likely outcome or assessment;
mark them as inferences, retain the relevant behavioral evidence and counterevidence, and avoid
turning lack of a statement into proof of its opposite. Do not infer sensitive traits from stereotypes.
Each row, inference, and constraint must cite only IDs present in the input, with direct support.
Do not include irrelevant encouragement. Keep the complete JSON under about 650 tokens.'''


def compile_table(api, question, context, encoder, cap=800):
    result=api.call(PROMPT,'Question: '+question+'\nDated memories:\n'+context['text'],
                    max_tokens=4096,json_output=True)
    table=result['parsed'];valid=set(context['source_ids'])
    table={key:[row for row in table.get(key,[]) if isinstance(row,dict)
                and isinstance(row.get('sources'),list) and row['sources']
                and all(isinstance(mid,str) and mid in valid for mid in row['sources'])]
           for key in ('rows','inferences','constraints')}
    def render():return 'Derived evidence table (check against original turns):\n'+json.dumps(table,ensure_ascii=False,separators=(',',':'))
    while len(encoder.encode(render()))>cap:
        key=next((k for k in ('constraints','inferences','rows') if table[k]),None)
        if key is None:raise ValueError('Evidence table cap is too small')
        table[key].pop()
    return dict(text=render(),tokens=len(encoder.encode(render())),table=table,
                api_hash=result['request_hash'],usage=result['usage'])


def retrieve_table(memory,question,budget,encoder,dense,api):
    initial=memory.retrieve(question,'hybrid_packed',budget,encoder,dense)
    state=compile_table(api,question,initial,encoder,min(800,budget//3))
    originals=memory.retrieve(question,'hybrid_packed',budget-state['tokens']-16,encoder,dense)
    text=state['text']+'\n\nOriginal turns:\n'+originals['text']
    # The compiler has read the same initial budget as the baseline. Record its
    # evidence separately from the originals retained by the final answer call.
    refs={mid for rows in state['table'].values() for row in rows for mid in row['sources']}
    return dict(text=text,tokens=len(encoder.encode(text)),facts=[],
                source_ids=list(dict.fromkeys(originals['source_ids']+sorted(refs))),
                original_source_ids=originals['source_ids'],compiler=state,
                compiler_input_tokens=initial['tokens'])
