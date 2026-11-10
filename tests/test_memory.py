import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from memory import evolve, Memory, locomo_sessions


def test_state_history_and_preferences():
    base = dict(subject='Alice', slot='residence', scope='', kind='state', single=True)
    facts = [dict(base, id='old', order=0), dict(base, id='new', order=1),
             dict(base, id='hobby', order=2, single=False)]
    result = evolve(facts)
    assert result[0]['next'] == ['new']
    assert result[1]['previous'] == ['old']
    assert result[2]['previous'] == []
    assert 'next' not in facts[0]  # Input ledger is immutable.


def test_retrieval_budget_and_exact_append_only_when_evolution_off():
    class Encoder:
        def encode(self, text):
            return list(text.encode())
    turns = [dict(id=str(i), speaker='Alice', text='city Paris' if i == 0 else 'city London',
                  date=str(i), session=str(i), order=i) for i in range(2)]
    facts = evolve([dict(id='f' + str(i), subject='Alice', slot='residence', scope='', kind='state',
                         single=True, order=i, observed_at=str(i), text=t['text'], sources=[t['id']])
                    for i, t in enumerate(turns)])
    m = Memory(dict(sessions=[dict(turns=turns)], facts=facts))
    before = m.retrieve('city', 'facts', 1000, Encoder())
    m.retrieve('city', 'evolved', 1000, Encoder())
    assert before == m.retrieve('city', 'facts', 1000, Encoder())
    assert m.retrieve('city', 'evolved', 100, Encoder())['tokens'] <= 100


def test_current_query_can_recover_newer_differently_worded_observation():
    class Encoder:
        def encode(self, text):
            return list(text.encode())
    turns = [dict(id='old', speaker='user', text='My current camera lens is a 50mm prime.',
                  date='2023-01-01', session='s0', order=0),
             dict(id='new', speaker='user', text='I upgraded to a 70-200mm zoom.',
                  date='2023-02-01', session='s1', order=1)]
    facts = [dict(id='f'+str(i), subject='user', slot='gear', kind='state', single=False,
                  order=i, observed_at=t['date'], text=t['text'], sources=[t['id']])
             for i,t in enumerate(turns)]
    data = dict(sessions=[dict(turns=turns)], facts=facts,
                tracks=[dict(entity='user', attribute='camera lenses', members=['f0','f1'])])
    m = Memory(data)
    result = m.retrieve('What is my current camera lens?', 'consolidated', 250, Encoder())
    assert 'new' in result['source_ids']
    assert result['tokens'] <= 250


def test_caption_protocol_keeps_search_queries_and_qa_out_of_writer_input():
    sample = dict(conversation=dict(session_1_date_time='2023-01-01', session_1=[
        dict(dia_id='D1:1', speaker='Alice', text='Look at this.',
             blip_caption='a cup with a dog face', query='SECRET_SEARCH_HINT')]),
        qa=[dict(question='SECRET_QUESTION', answer='SECRET_ANSWER')])
    import json
    original = json.dumps(locomo_sessions(sample))
    captions = json.dumps(locomo_sessions(sample, True))
    assert 'a cup with a dog face' not in original
    assert 'a cup with a dog face' in captions
    assert 'SECRET_' not in original + captions
