"""Isolated SiliconFlow backend for phase-3 experiments; v2 OpenAI runs stay unchanged."""
import json
import os
import random
import threading
import time
import requests
from api import ROOT,atomic_json,digest

MODEL='deepseek-ai/DeepSeek-V4.1-Flash'
URL='https://api.siliconflow.cn/v1/chat/completions'
BALANCE_EXHAUSTED=threading.Event()


class SiliconFlowBalanceError(RuntimeError):
    pass


class SiliconFlowAPI:
    def __init__(self,enable_thinking=False):
        self.model=MODEL
        self.enable_thinking=enable_thinking
        self.cache=ROOT/'runs/phase3/api_cache';self.cache.mkdir(parents=True,exist_ok=True)
        self.key=os.environ.get('SILICONFLOW_API_KEY') or (ROOT/'.private/siliconflow_key').read_text().strip()
        self.local=threading.local()

    def call(self,instructions,content,max_tokens=4096,json_output=False,effort=None):
        body=dict(model=self.model,messages=[dict(role='system',content=instructions),dict(role='user',content=content)],
                  max_tokens=max_tokens,temperature=0,enable_thinking=self.enable_thinking)
        if json_output:
            body['response_format']=dict(type='json_object')
            body['messages'][1]['content']='Return JSON.\n'+content
        signature=digest(dict(endpoint=URL,body=body))
        path=self.cache/(signature+'.json')
        if path.exists():return json.loads(path.read_text())
        if BALANCE_EXHAUSTED.is_set():raise SiliconFlowBalanceError('SiliconFlow balance is insufficient; no further requests sent')
        if not hasattr(self.local,'session'):
            s=requests.Session();s.trust_env=False
            proxy=os.environ.get('MEMORY_PROXY','http://127.0.0.1:41227')
            s.proxies={'http':proxy,'https':proxy};self.local.session=s
        start=time.time()
        for attempt in range(6):
            if BALANCE_EXHAUSTED.is_set():raise SiliconFlowBalanceError('SiliconFlow balance is insufficient; no further requests sent')
            try:
                r=self.local.session.post(URL,headers={'Authorization':'Bearer '+self.key},json=body,timeout=180)
            except (requests.Timeout,requests.ConnectionError):
                if attempt==5:raise RuntimeError('SiliconFlow connection failed') from None
                time.sleep(min(10,2**attempt));continue
            if (r.status_code==429 or r.status_code>=500) and attempt<5:
                time.sleep(min(10,2**attempt)+random.random());continue
            if r.status_code==402:
                BALANCE_EXHAUSTED.set()
                raise SiliconFlowBalanceError('SiliconFlow HTTP 402: account balance is insufficient')
            if r.status_code!=200:
                raise RuntimeError('SiliconFlow HTTP {}: {}'.format(r.status_code,r.text[:500].replace(self.key,'[REDACTED]')))
            data=r.json();choice=data['choices'][0]
            if choice.get('finish_reason')=='length' and body['max_tokens']==max_tokens:
                body['max_tokens']*=2;continue
            if choice.get('finish_reason')!='stop':raise RuntimeError('Incomplete SiliconFlow output: '+str(choice.get('finish_reason')))
            value=choice['message'].get('content','')
            if not value.strip():raise RuntimeError('SiliconFlow returned empty text')
            result=dict(text=value,model=data.get('model'),usage=data.get('usage'),request_hash=signature,
                        elapsed_seconds=time.time()-start,provider='SiliconFlow',enable_thinking=self.enable_thinking)
            if body['max_tokens']!=max_tokens:result['served_max_tokens']=body['max_tokens']
            if json_output:result['parsed']=json.loads(value)
            atomic_json(path,result);return result
