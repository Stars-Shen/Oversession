"""Deterministic calendar offsets beside original, source-attributed dialogue."""
from datetime import datetime,timedelta
import re

TIME_QUERY=re.compile(r'\bwhen\b|\bhow (?:long|old)\b|\b(?:before|after|year|month|week|date|since)\b',re.I)
DAYS=['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
NUMBERS={name:i for i,name in enumerate(['one','two','three','four','five','six','seven','eight','nine','ten'],1)}
NUMBERS['a']=1
EXPRESSION=re.compile(r'\b(?:yesterday|tomorrow|last (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|month|year)|(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|a) (?:days?|weeks?|months?|years?) ago)\b',re.I)


def annotate(text,date):
    match=re.search(r'\bon (\d{1,2} [A-Za-z]+, \d{4})',date)
    if not match:return text,[]
    anchor=datetime.strptime(match[1],'%d %B, %Y').date();notes=[]
    def resolve(match):
        phrase=match[0];value=phrase.lower();parts=value.split();precision='day'
        if value in ('yesterday','tomorrow'):result=anchor+timedelta(days=-1 if value=='yesterday' else 1)
        elif parts[0]=='last' and parts[1] in DAYS:
            offset=(anchor.weekday()-DAYS.index(parts[1]))%7 or 7;result=anchor-timedelta(days=offset)
        else:
            amount=1 if parts[0]=='last' else (int(parts[0]) if parts[0].isdigit() else NUMBERS[parts[0]])
            unit=parts[1].rstrip('s')
            if unit in ('day','week'):result=anchor-timedelta(days=amount*(7 if unit=='week' else 1))
            elif unit=='month':
                year,month=divmod(anchor.year*12+anchor.month-1-amount,12)
                result='{} {}'.format(datetime(year,month+1,1).strftime('%B'),year);precision='month'
            else:result=str(anchor.year-amount);precision='year'
        if precision=='day':result='{} {}'.format(result.day,result.strftime('%B %Y'))
        notes.append(dict(phrase=phrase,reference=result,precision=precision,anchor=anchor.isoformat()))
        return phrase+' [calendar offset: '+result+']'
    # Caption text is generated image metadata, not a speaker's relative-time claim.
    speech,separator,caption=text.partition(' [Shared image caption (automatic):')
    return EXPRESSION.sub(resolve,speech)+separator+caption,notes


def context(memory,source_ids,encoder,budget=3000):
    selected=list(source_ids)
    while True:
        parts=[];notes=[];session=None
        for mid in sorted(selected,key=memory.turn_pos.get):
            turn=memory.by_turn[mid]
            if turn['session']!=session:
                session=turn['session'];parts.append('Session {} | {}'.format(session,turn['date']))
            text,anchors=annotate(turn['text'],turn['date'])
            parts.append('[{}] {}: {}'.format(mid,turn['speaker'],text))
            notes.extend(dict(note,source=mid) for note in anchors)
        text='\n'.join(parts);size=len(encoder.encode(text))
        if size<=budget:return dict(text=text,tokens=size,source_ids=selected,anchors=notes)
        selected.pop()  # Original source list is in descending retrieval priority.
