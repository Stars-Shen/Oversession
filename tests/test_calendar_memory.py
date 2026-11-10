import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from calendar_memory import annotate


def test_calendar_offsets_keep_precision_and_original_words():
    text,notes=annotate('Yesterday, last Monday, last Friday, two weeks ago, last month, and two years ago.',
                        '1:56 pm on 8 May, 2023')
    assert [n['reference'] for n in notes]==['7 May 2023','1 May 2023','5 May 2023','24 April 2023','April 2023','2021']
    assert [n['precision'] for n in notes][-2:]==['month','year']
    assert 'Yesterday [calendar offset: 7 May 2023]' in text


def test_caption_unknown_offsets_and_unknown_date_are_not_promoted():
    original='Several days ago. [Shared image caption (automatic): yesterday at a beach]'
    assert annotate(original,'1:56 pm on 8 May, 2023')==(original,[])
    assert annotate('Yesterday','unknown')==('Yesterday',[])
