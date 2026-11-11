"""Opt-in Flex transport for the existing Luna workflow; baseline API unchanged."""
import time
import requests
from api import API,ROOT,atomic_json,digest


def request_body(model,instructions,content,max_tokens=4096,json_output=False,effort='low'):
    body=dict(model=model,instructions=instructions,input=content,max_output_tokens=max_tokens,
        store=False,reasoning={'effort':effort})
    if json_output:
        body['text']={'format':{'type':'json_object'}};body['input']='Return JSON.\n'+content
    return body


def luna_cost(usage,flex):
    """2026-11-11 public USD estimate; output already includes reasoning tokens."""
    details=usage.get('input_tokens_details') or {}
    read=details.get('cached_tokens',0);write=details.get('cache_write_tokens',0)
    ordinary=usage['input_tokens']-read-write
    assert ordinary>=0
    multiplier=.5 if flex else 1.;long=usage['input_tokens']>272000
    return multiplier*((ordinary*.1+read*.01+write*.125)*(2 if long else 1)+usage['output_tokens']*.5*(1.5 if long else 1))/1e6


class FlexAPI(API):
    def call(self,instructions,content,max_tokens=4096,json_output=False,effort='low'):
        assert self.model=='gpt-6-luna','This price-audited transport is scoped to the existing Luna model.'
        body=request_body(self.model,instructions,content,max_tokens,json_output,effort)
        signature=digest(body);folder=ROOT/'runs/api_flex_cache';path=folder/(signature+'.json')
        # Transport tier does not change the requested model/instructions/input.
        for existing in (self.cache/(signature+'.json'),path):
            if existing.exists():
                import json
                return dict(json.loads(existing.read_text()),new_paid_call=False,estimated_new_cost_usd=0.,cache_source='legacy' if existing.parent==self.cache else 'flex')
        sent=dict(body,service_tier='flex');start=time.time()
        with requests.Session() as session:
            session.trust_env=False
            import os
            proxy=os.environ.get('MEMORY_PROXY','http://127.0.0.1:41227');session.proxies={'http':proxy,'https':proxy}
            # No automatic retries or Standard fallback: ambiguous timeouts must
            # not silently duplicate paid requests, or increase the service rate.
            response=session.post('https://api.openai.com/v1/responses',
                headers={'Authorization':'Bearer '+self.key},json=sent,timeout=(30,900))
        if response.status_code!=200:
            raise RuntimeError('OpenAI Flex HTTP {}: {}'.format(response.status_code,response.text[:400].replace(self.key,'[REDACTED]')))
        data=response.json();text='\n'.join(c['text'] for item in data.get('output',[])
            for c in item.get('content',[]) if c.get('type')=='output_text')
        complete=data.get('status')=='completed' and bool(text.strip());served=data.get('service_tier')
        result=dict(text=text if complete else 'No information available',model=data.get('model'),usage=data['usage'],
            complete=complete,incomplete_details=data.get('incomplete_details'),raw_output_text=text,
            request_hash=signature,served_request_hash=digest(sent),requested_service_tier='flex',
            served_service_tier=served,elapsed_seconds=time.time()-start,new_paid_call=True,
            estimated_new_cost_usd=luna_cost(data['usage'],served=='flex'),price_date='2026-11-11')
        if json_output and complete:
            import json
            result['parsed']=json.loads(text)
        atomic_json(path,result)
        return result
