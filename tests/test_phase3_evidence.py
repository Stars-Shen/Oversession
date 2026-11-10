import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evidence_table import compile_table
from cross_rank import retrieve_ranked
from memory import Memory


class Encoder:
    def encode(self,text):return list(text.encode())


def test_compiler_rejects_nonexistent_witnesses_even_in_inferences():
    class FakeAPI:
        def call(self,*args,**kwargs):
            return dict(parsed=dict(rows=[dict(value='valid',sources=['a']),dict(value='invented',sources=['z'])],
                                    inferences=[dict(claim='unsupported',sources=['a','z'])]),
                        request_hash='fake',usage={})
    state=compile_table(FakeAPI(),'What happened?',dict(text='[a] original',source_ids=['a']),Encoder(),800)
    assert state['table']['rows']==[dict(value='valid',sources=['a'])]
    assert state['table']['inferences']==[]


def test_ranking_cannot_split_a_joint_dialogue_window_to_fit_budget():
    turns=[dict(id=str(i),speaker='Alice',text='evidence '+str(i),date='2023',session='s0',order=i) for i in range(3)]
    memory=Memory(dict(sessions=[dict(turns=turns)],facts=[]))
    ranking=[dict(sources=['0','1'],score=.9),dict(sources=['1','2'],score=.8)]
    result=retrieve_ranked(memory,ranking,1000,Encoder())
    assert result['source_ids']==['0','1','2']
    assert result['text'].count('[1]')==1
    too_small=retrieve_ranked(memory,ranking,20,Encoder())
    assert too_small['source_ids']==[]


def test_insufficient_balance_stops_http_calls_across_reader_instances(tmp_path,monkeypatch):
    import threading
    import pytest
    import siliconflow
    monkeypatch.setattr(siliconflow,'ROOT',tmp_path)
    monkeypatch.setattr(siliconflow,'BALANCE_EXHAUSTED',threading.Event())
    monkeypatch.setenv('SILICONFLOW_API_KEY','test-credential')
    calls=[]
    class Session:
        def post(self,*args,**kwargs):
            calls.append(1)
            return type('Response',(),dict(status_code=402))()
    first=siliconflow.SiliconFlowAPI();second=siliconflow.SiliconFlowAPI(enable_thinking=True)
    first.local.session=Session();second.local.session=Session()
    with pytest.raises(siliconflow.SiliconFlowBalanceError):first.call('test','first')
    with pytest.raises(siliconflow.SiliconFlowBalanceError):second.call('test','second')
    assert len(calls)==1
