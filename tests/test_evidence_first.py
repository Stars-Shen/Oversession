import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evidence_first_study import parse_answer


def test_answer_parser_preserves_answer_and_rejects_invented_sources():
    text='```json\n{"observations":[{"s":["D1:2"],"v":"A real observation"}],"answer":"Paris"}\n```'
    assert parse_answer(text,{'D1:2'})[0]=='Paris'
    with pytest.raises(ValueError):parse_answer(text,{'D1:3'})


def test_abstention_can_have_no_witness_but_a_factual_answer_cannot():
    assert parse_answer('{"observations":[],"answer":"No information available"}',set())[0]=='No information available'
    with pytest.raises(ValueError):parse_answer('{"observations":[],"answer":"Paris"}',set())
