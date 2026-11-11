import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from memory import Memory
from joint_selection import memory_index, selected_context


class Encoder:
    def encode(self, text):
        return list(text.encode())


def memory():
    turns = [dict(id='D' + str(i), speaker='Alice', date=str(i), session='s' + str(i // 2),
                  text='evidence ' + str(i), order=i // 2) for i in range(4)]
    return Memory(dict(sessions=[dict(turns=turns)],
        facts=[dict(id='f', subject='Alice', slot='event', text='two-root fact', sources=['D1', 'D2'])],
        qa=[dict(question='SECRET_QUESTION', answer='SECRET_GOLD')]))


def test_index_allowlist_and_residuals_preserve_exact_original_source_map():
    m = memory(); original = copy.deepcopy(m.data); index = memory_index(m)
    assert 'SECRET' not in json.dumps(index)
    assert index['roots'] == dict(F0=['D1', 'D2'], R0=['D0'], R1=['D3'])
    assert m.data == original


def test_joint_roots_and_local_context_are_complete_and_chronological():
    m = memory(); index = memory_index(m)
    context = selected_context(m, index, dict(ids=['F0', 'F0', 'R1']), 2000, Encoder())
    assert context['roots'] == ['D1', 'D2', 'D3']
    assert context['source_ids'] == ['D0', 'D1', 'D2', 'D3']
    assert context['selected_aliases'] == ['F0', 'R1']
    assert context['text'].index('[D1]') < context['text'].index('[D2]')
    assert 'two-root fact' not in context['text']  # Generated index text is not answer context.


def test_invalid_or_overbudget_selection_never_returns_partial_evidence():
    m = memory(); index = memory_index(m)
    for ids in ([], ['F0', 'UNKNOWN'], ['F0', 2]):
        assert selected_context(m, index, dict(ids=ids), 2000, Encoder()) is None
    assert selected_context(m, index, dict(ids=['F0']), 20, Encoder()) is None
    assert selected_context(m, index, dict(ids=['R1']), 2000, Encoder())['source_ids'] == ['D2', 'D3']
