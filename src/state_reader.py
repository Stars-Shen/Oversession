"""Compile a request-specific evidence state without access to labels or answers."""
import json

PROMPT = '''Compile a concise evidence state for a question over dated conversation memory.
Do not answer the question and do not produce a reasoning narrative. Return JSON with fields:
{"operation":"lookup|latest|count|sum|time_difference|preference|inference",
 "target":"entity, attribute and scope requested",
 "time_scope":"relevant date range or null",
 "observations":[{"sources":["source ID"],"date":"event date or observed date with uncertainty",
                  "fact":"concise, specific evidence including exact numbers/names",
                  "role":"applicable|historical|constraint|uncertain"}],
 "application_rules":["short evidence-use constraints"],"missing":"missing evidence or null"}.
For latest/current requests, compare observations of the SAME attribute and preserve older values
as historical. A new unrelated fact does not invalidate an old fact. Use event time when available;
report time uncertainty rather than interpreting every mention as a new event.
For count/sum, preserve each distinct contributing event, distinguish actual vs planned events,
and avoid counting repeated mentions of the same event twice. Preserve each amount and date.
For preferences, retain specific personal successes, constraints and goals needed to tailor advice.
Every observation must cite IDs present in the input. Do not invent facts or use outside personal
knowledge. A missing exact statement does not preclude an explicitly requested inference.
Keep the whole JSON compact, ideally under 700 tokens.''' 


def compile_state(api, question, question_date, context, budget, encoder):
    content = 'Current date: ' + question_date + '\nQuestion: ' + question + '\nMemories:\n' + context['text']
    result = api.call(PROMPT, content, max_tokens=4096, json_output=True)
    state = result['parsed']
    valid = set(context['source_ids']) | set(context['facts'])
    state['observations'] = [o for o in state.get('observations', [])
                             if o.get('sources') and set(o['sources']) <= valid]
    # Keep the final reader's total evidence budget fixed. Full compiler input is cached.
    while state.get('observations') and len(encoder.encode(json.dumps(state))) > min(1000, budget // 3):
        state['observations'].pop()
    text = json.dumps(state, ensure_ascii=False)
    return dict(text=text, tokens=len(encoder.encode(text)), api_hash=result['request_hash'], usage=result['usage'])
