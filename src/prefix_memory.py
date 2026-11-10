"""Question-independent raw history and causal prefix reuse for a frozen decoder."""


def raw_history(memory):
    parts=[];previous=None;ids=[]
    for session in memory['sessions']:
        for turn in session['turns']:
            if turn['session']!=previous:
                parts.append('Session '+turn['session']+' | '+turn['date']);previous=turn['session']
            parts.append('['+turn['id']+'] '+turn['speaker']+': '+turn['text']);ids.append(turn['id'])
    return '\n'.join(parts),ids


def prompt_ids(tokenizer,row):
    return tokenizer.apply_chat_template([dict(role='system',content=row['system']),
        dict(role='user',content='Memories:\n'+row['context']+'\n\nQuestion: '+row['question'])],
        tokenize=True,add_generation_prompt=True)


def memory_prefix_ids(tokenizer,row):
    marker='__MEMORY_QUESTION_BOUNDARY__'
    text=tokenizer.apply_chat_template([dict(role='system',content=row['system']),
        dict(role='user',content='Memories:\n'+row['context']+'\n\nQuestion: '+marker)],
        tokenize=False,add_generation_prompt=True)
    assert text.count(marker)==1
    # Leave the boundary token uncached so whitespace can merge with the first
    # question token. This prefix depends only on the memory and instructions.
    return tokenizer.encode(text.split(marker)[0],add_special_tokens=False)[:-1]


def prefill(model,ids):
    import torch
    with torch.inference_mode():
        return model(ids,attention_mask=torch.ones_like(ids),use_cache=True,logits_to_keep=1).past_key_values


def answer_from_prefix(model,ids,cache,prefix_length,generation_config):
    import torch
    assert cache.get_seq_length()==prefix_length
    with torch.inference_mode():
        result=model.generate(ids,attention_mask=torch.ones_like(ids),past_key_values=cache,
            generation_config=generation_config,return_dict_in_generate=True,
            do_sample=False,temperature=1.,top_p=1.,top_k=50,logits_to_keep=1)
    cache=result.past_key_values;cache.crop(prefix_length)
    return result.sequences,cache
