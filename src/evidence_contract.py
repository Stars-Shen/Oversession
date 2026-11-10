"""Source-membership contract shared by generation and evaluation."""
import json


def parse_answer(text,valid_sources):
    value=json.loads(text[text.find('{'):text.rfind('}')+1])
    answer=value['answer']
    if not isinstance(answer,str) or not answer.strip():raise ValueError('Missing answer string')
    observations=value['observations']
    if not isinstance(observations,list):raise ValueError('Missing observation list')
    for item in observations:
        if not isinstance(item,dict) or not isinstance(item.get('s'),list) or not item['s']:
            raise ValueError('Observation has no sources')
        if not all(isinstance(mid,str) and mid in valid_sources for mid in item['s']):
            raise ValueError('Observation cites a source outside the supplied context')
        if not isinstance(item.get('v'),str) or not item['v'].strip():raise ValueError('Empty observation')
    if not observations and answer.strip().lower().rstrip('.')!='no information available':
        raise ValueError('Unsupported answer has no observations')
    return answer.strip(),value


def source_schema(sources):
    return dict(type='object',properties={
        'observations':dict(type='array',maxItems=3,items=dict(type='object',properties={
            's':dict(type='array',minItems=1,items=dict(type='string',enum=sources)),
            'v':dict(type='string'),'t':dict(type='string')},required=['s','v','t'],additionalProperties=False)),
        'operation':dict(type='string',enum=['lookup','compare','count','date','inference']),
        'check':dict(type='string'),'answer':dict(type='string')},
        required=['observations','operation','check','answer'],additionalProperties=False)
