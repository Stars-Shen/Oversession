"""Integrity checks for relations; these do not certify generated entailment."""
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from memory import Memory
from relational_memory import accept_cards, active_cards, build_ledger, RelationMemory


class Encoder:
    def encode(self, text):
        return list(text.encode())


def fixture():
    texts = ['Alice recommended the book Atlas.',
             'Bob is reading the book Alice recommended.',
             'Bob finished that recommended book.']
    turns = [dict(id='D' + str(i), speaker='Bob', text=text, session='s' + str(i),
                  date='2023-01-0' + str(i + 1), order=i) for i, text in enumerate(texts)]
    return dict(sessions=[dict(id=t['session'], date=t['date'], turns=[t]) for t in turns],
                facts=[dict(id='f' + str(i), subject='Bob', slot='books', text=t['text'],
                            session=t['session'], observed_at=t['date'], sources=[t['id']], order=i)
                       for i, t in enumerate(turns)], qa=[dict(answer='SECRET_GOLD')])


def proposal(**changes):
    return dict(dict(subject='Bob', relation='reading', scope='Atlas', text='Bob is reading Atlas.',
                     support_groups=[['f0', 'f1']], revises=None), **changes)


def test_future_unknown_and_incomplete_alternative_proofs_are_rejected():
    data = fixture(); session = data['sessions'][1]
    for groups, expected in [([['f0', 'f2']], 'future_fact'),
                             ([['f0', 'missing']], 'unknown_fact'),
                             ([['f0', 'f1'], ['f1']], 'invalid_group_size')]:
        cards, rejected = accept_cards([proposal(support_groups=groups)], data, session, [])
        assert not cards and rejected[0]['reason'] == expected


def test_version_updates_are_local_and_inputs_remain_immutable():
    data = fixture(); before = copy.deepcopy(data)
    cards, rejected = accept_cards([proposal()], data, data['sessions'][1], [])
    assert not rejected
    old = copy.deepcopy(cards)
    bad = proposal(scope='another book', revises=cards[0]['id'], support_groups=[['f0', 'f2']])
    assert accept_cards([bad], data, data['sessions'][2], cards)[1][0]['reason'] == 'nonlocal_revision'
    good = proposal(text='Bob was reading Atlas and has now finished it.',
                    revises=cards[0]['id'], support_groups=[['f0', 'f1', 'f2']])
    new, rejected = accept_cards([good], data, data['sessions'][2], cards)
    assert not rejected and cards == old and data == before
    assert active_cards(cards + new) == new
    assert len(cards + new) == 2


def test_retrieval_packs_a_whole_and_group_not_the_union_of_or_groups():
    data = fixture(); m = Memory(data)
    # Construct a ledger directly to test retrieval independently of evolution.
    cards = [dict(proposal(support_groups=[['f0', 'f1'], ['f0', 'f2']]),
                  id='relation', observed_at='2023-01-03')]
    r = RelationMemory(m, cards)
    dense = dict(raw=[(i, 1.) for i in range(3)], facts=[(i, 1.) for i in range(3)])
    baseline = m.retrieve_packed('What book is Bob reading?', 2000, Encoder(), dense)
    assert r.retrieve('What book is Bob reading?', 2000, Encoder(), dense, enabled=False) == baseline
    result = r.retrieve('What book is Bob reading?', 2000, Encoder(), dense)
    assert result['relations'][0]['support_group'] in [['f0', 'f1'], ['f0', 'f2']]
    assert len(result['relations'][0]['sources']) == 2
    assert set(result['relations'][0]['sources']) <= set(result['source_ids'])
    assert result['tokens'] <= 2000
    small = r.retrieve('What book is Bob reading?', 100, Encoder(), dense)
    assert small == m.retrieve_packed('What book is Bob reading?', 100, Encoder(), dense)


def test_builder_passes_only_a_chronological_fact_allowlist(tmp_path):
    class FakeAPI:
        model = 'gpt-6-luna'
        calls = []

        def call(self, instructions, content, **kwargs):
            self.calls.append(json.loads(content))
            assert 'SECRET_GOLD' not in content
            return dict(parsed=dict(cards=[]), complete=True, new_paid_call=False,
                        estimated_new_cost_usd=0., request_hash='fake', usage={})

    api = FakeAPI(); data = fixture(); before = copy.deepcopy(data)
    path = tmp_path / 'ledger.json'; build_ledger(data, api, path)
    assert len(api.calls) == 2
    assert [f['id'] for f in api.calls[0]['past']] == ['f0']
    assert [f['id'] for f in api.calls[0]['incoming']] == ['f1']
    assert data == before
    build_ledger(data, api, path)
    assert len(api.calls) == 2  # A completed prefix is not regenerated.
