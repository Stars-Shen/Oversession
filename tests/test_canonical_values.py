import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from canonical_values import retrieve
from memory import Memory


def test_canonical_values_deduplicate_joint_support_keep_raw_fallback_and_preserve_baseline():
    class Encoder:
        def encode(self,text):return list(text.encode())
    turns=[dict(id='D1:'+str(i),speaker='Alice',text=text,date='2023-01-01',session='S1',order=0)
           for i,text in enumerate(['I moved yesterday.','To Paris.','My favorite color is blue.'])]
    facts=[dict(id='f0',subject='Alice',slot='residence',kind='state',single=True,
                text='Alice moved to Paris on 31 December 2022.',sources=['D1:0','D1:1'],
                observed_at='2023-01-01',session='S1',order=0)]
    memory=Memory(dict(sessions=[dict(turns=turns)],facts=facts));snapshot=copy.deepcopy(memory.data)
    dense=dict(raw=[(0,1),(1,.9),(2,.1)],facts=[(0,1)])
    before=memory.retrieve_packed('Alice moved Paris',1000,Encoder(),dense)
    result=retrieve(memory,'Alice moved Paris',1000,Encoder(),dense)
    assert result['text'].count('Alice moved to Paris')==1
    assert 'favorite color is blue' in result['text']
    assert set(result['source_ids'])=={'D1:0','D1:1','D1:2'}
    assert result['original_source_ids']==['D1:2']
    assert result['fact_ids']==['f0'] and result['tokens']<=1000
    assert memory.data==snapshot
    assert memory.retrieve_packed('Alice moved Paris',1000,Encoder(),dense)==before
    assert retrieve(memory,'Alice moved Paris',60,Encoder(),dense)['tokens']<=60
