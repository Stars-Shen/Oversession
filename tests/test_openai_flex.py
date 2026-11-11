import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from openai_flex import FlexAPI,luna_cost,model_cost,request_body
from api import digest


def test_legacy_cache_avoids_paid_request(tmp_path):
    body=request_body('gpt-6-luna','system','input',768,False,'low')
    expected=dict(model='gpt-6-luna',instructions='system',input='input',max_output_tokens=768,
        store=False,reasoning={'effort':'low'})
    assert body==expected
    (tmp_path/(digest(body)+'.json')).write_text(json.dumps(dict(text='saved',usage={})))
    client=object.__new__(FlexAPI);client.model='gpt-6-luna';client.cache=tmp_path
    with patch('openai_flex.requests.Session',side_effect=AssertionError('paid call on cache hit')):
        result=client.call('system','input',768)
    assert result['text']=='saved' and result['estimated_new_cost_usd']==0 and not result['new_paid_call']


def test_reprice_partitions_input_and_counts_reasoning_only_once():
    usage=dict(input_tokens=1000,input_tokens_details=dict(cached_tokens=200,cache_write_tokens=600),
        output_tokens=100,output_tokens_details=dict(reasoning_tokens=80))
    standard=(200*.1+200*.01+600*.125+100*.5)/1e6
    assert luna_cost(usage,False)==standard
    assert luna_cost(usage,True)==standard/2


def test_sol_has_its_own_cached_read_rate():
    usage=dict(input_tokens=1000,input_tokens_details=dict(cached_tokens=200,cache_write_tokens=600),output_tokens=100)
    assert model_cost(usage,'gpt-6.1-sol',True)==(200*2+200*.1+600*2.5+100*10)/2e6


def test_only_luna_is_allowed_before_request_setup():
    client=object.__new__(FlexAPI);client.model='gpt-6.1-sol'
    with pytest.raises(AssertionError,match='gpt-6-luna only'):
        client.call('system','input')
