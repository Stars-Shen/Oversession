"""Causality and attribution contracts, not a generated fact-quality metric."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from turn_writer import inputs, observations, assemble


def memory():
    turns = [dict(id='D' + str(i), speaker='Alice', text=text, date=str(i), session='s' + str(i // 2), order=i // 2)
             for i, text in enumerate(['I live in Paris.', 'I like the city.', 'I moved to Berlin.', 'FUTURE_SECRET'])]
    return dict(sessions=[dict(id='s' + str(i), date=str(i), turns=turns[i*2:i*2+2]) for i in range(2)],
                facts=[dict(id='f0', subject='Alice', slot='residence', scope='', kind='state', single=True,
                            text='Alice lives in Paris.', session='s0', observed_at='0', sources=['D0']),
                       dict(id='f1', subject='Alice', slot='future', text='FUTURE_SECRET', session='s1', sources=['D3'])],
                qa=[dict(answer='GOLD_SECRET')])


def fact(**changes):
    return dict(dict(subject='Alice', slot='residence', scope='', kind='state', single=True,
                     text='Alice lives in Berlin.', event_date=None, sources=['D2']), **changes)


def test_writer_inputs_exclude_future_and_qa_but_include_prior_session_sources():
    data = memory(); before = copy.deepcopy(data); rows = inputs(data); current = rows[2]
    assert 'FUTURE_SECRET' not in json.dumps(current)
    assert all('GOLD_SECRET' not in json.dumps(r) for r in rows)
    assert current['payload']['historical_facts'][0]['id'] == 'f0'
    assert current['payload']['historical_sources'][0]['id'] == 'D0'
    assert [t['id'] for t in current['payload']['preceding']] == ['D0', 'D1']
    assert data == before


def test_context_only_claims_and_future_citations_are_not_accepted():
    row = inputs(memory())[2]
    facts, rejected = observations(row, [fact(sources=['D0']), fact(sources=['D2', 'D3']), fact(sources=['f0', 'D2'])])
    assert not facts
    assert [r['reason'] for r in rejected] == ['no_target_evidence', 'invisible_source', 'invisible_source']
    facts, rejected = observations(row, [fact(sources=['D0', 'D2'])])
    assert not rejected and facts[0]['origin_turn'] == 'D2' and facts[0]['sources'] == ['D0', 'D2']


def test_turn_local_versions_preserve_historical_observations_and_raw_data():
    data = memory(); before = copy.deepcopy(data); rows = inputs(data); extracted = []
    for i, row in enumerate(rows):
        proposals = [fact(text='Alice lives in Paris.' if i == 0 else 'Alice lives in Berlin.', sources=[row['target_id']])] if i in (0, 2) else []
        facts, _ = observations(row, proposals)
        extracted.append(dict(target_id=row['target_id'], order=i, facts=facts, complete=True))
    result = assemble(data, list(reversed(extracted)))
    assert result['sessions'] == before['sessions'] and data == before
    assert result['facts'][0]['next'] == [result['facts'][1]['id']]
    assert result['facts'][1]['previous'] == [result['facts'][0]['id']]
    assert result['facts'][0]['text'] == 'Alice lives in Paris.'
