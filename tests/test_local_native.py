from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from local_native import parse_completion


def test_only_final_answer_is_exposed_to_the_scorer():
    words={1:'private reasoning',2:'</think>',3:'Paris',4:''}
    decode=lambda ids:' '.join(words[i] for i in ids)
    row=parse_completion([1,2,3,4],2,True,decode)
    assert row['hypothesis']=='Paris' and row['reasoning']=='private reasoning'
    assert row['reasoning_tokens']==1 and row['closed_reasoning']
    assert parse_completion([1,2,1,2,3,4],2,True,decode)['hypothesis']=='Paris'
    assert parse_completion([3,4],2,False,decode)['hypothesis']=='Paris'


def test_unfinished_reasoning_or_missing_final_is_not_an_answer():
    decode=lambda ids:''.join({1:'Paris is a possibility',2:'',3:''}[i] for i in ids)
    for tokens in ([1],[1,2,3]):
        row=parse_completion(tokens,2,True,decode)
        assert row['hypothesis']=='No information available' and not row['closed_reasoning']
