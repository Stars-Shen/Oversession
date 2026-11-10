"""Training-only positive-source coverage from one actual decoder attention row."""
import re


def source_char_spans(context):
    boundaries=list(re.finditer(r'^\[D\d+:\d+\]|^Session ',context,re.M))
    sources={}
    for index,marker in enumerate(boundaries):
        match=re.match(r'\[(D\d+:\d+)\] [^:\n]+: ',context[marker.start():])
        if not match:continue
        mid=match[1];assert mid not in sources
        end=boundaries[index+1].start() if index+1<len(boundaries) else len(context)
        sources[mid]=(marker.start()+match.end(),end)
    return sources


def encoded_prompt(tokenizer,row):
    text=tokenizer.apply_chat_template([dict(role='system',content=row['system']),
        dict(role='user',content='Memories:\n'+row['context']+'\n\nQuestion: '+row['question'])],
        tokenize=False,add_generation_prompt=True)
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    offset=text.index(row['context']);spans={}
    for mid,(start,end) in source_char_spans(row['context']).items():
        indices=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>a and a>=offset+start and b<=offset+end]
        assert indices;spans[mid]=(min(indices),max(indices)+1)
    assert set(spans)==set(row['source_ids'])
    return encoded['input_ids'],spans


def coverage_loss(attention,spans,positives):
    import torch
    # Different heads may cover different required sources. Unannotated turns
    # are not assigned hard negative labels; other heads remain unconstrained.
    mass=torch.stack([attention[:,a:b].sum(-1) for mid,(a,b) in spans.items() if mid in positives],dim=-1)
    return -mass.max(dim=0).values.clamp_min(1e-8).log().mean()


class SourceAttention:
    def __init__(self,model):
        self.layer=model.model.layers[-1].self_attn;self.state={}
        self.handles=[self.layer.register_forward_pre_hook(self.before,with_kwargs=True),
                      self.layer.q_norm.register_forward_hook(self.query),
                      self.layer.k_norm.register_forward_hook(self.key)]
    def before(self,module,args,kwargs):
        self.state['position']=kwargs['position_embeddings']
    def query(self,module,args,output):self.state['q']=output
    def key(self,module,args,output):self.state['k']=output
    def attention(self,index):
        import torch
        cos,sin=self.state['position'];q=self.state['q'][:,index:index+1].transpose(1,2).float()
        k=self.state['k'][:,:index+1].transpose(1,2).float()
        def rotate(x):
            left,right=x.chunk(2,dim=-1);return torch.cat((-right,left),dim=-1)
        q=q*cos[:,None,index:index+1].float()+rotate(q)*sin[:,None,index:index+1].float()
        k=k*cos[:,None,:index+1].float()+rotate(k)*sin[:,None,:index+1].float()
        k=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
        return ((q@k.transpose(-1,-2))*self.layer.scaling).softmax(-1)[0,:,0]
    def close(self):
        for handle in self.handles:handle.remove()
        self.state.clear()
