import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires the existing torch environment')
class ReaderAdapterTests(unittest.TestCase):
    def make_model(self):
        from transformers import Qwen3Config,Qwen3ForCausalLM
        return Qwen3ForCausalLM(Qwen3Config(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8,vocab_size=32,tie_word_embeddings=True)).eval()

    def test_zero_initialized_adapter_preserves_logits_and_freezes_original_weights(self):
        import torch
        from reader_adapter import attach,answer_loss
        torch.manual_seed(1);model=self.make_model();ids=torch.tensor([[1,2,3,4,5]])
        with torch.no_grad():before=model(ids).logits.clone()
        snapshot={k:p.detach().clone() for k,p in model.named_parameters()}
        layers=attach(model,rank=2,alpha=4)
        with torch.no_grad():after=model(ids).logits
        self.assertTrue(torch.equal(before,after))
        answer_loss(model,ids,2).backward()
        self.assertTrue(any(layer.b.grad.abs().sum()>0 for layer in layers.values()))
        for name,p in model.named_parameters():
            if name.endswith(('.a','.b')):continue
            original=name.replace('.q_proj.base.','.q_proj.').replace('.v_proj.base.','.v_proj.')
            self.assertTrue(torch.equal(p,snapshot[original]));self.assertIsNone(p.grad)

    def test_sliced_answer_loss_matches_full_masked_causal_loss(self):
        import torch
        from reader_adapter import answer_loss
        torch.manual_seed(2);model=self.make_model();ids=torch.tensor([[1,2,3,4,5]])
        labels=ids.clone();labels[:,:3]=-100
        full=model(ids,labels=labels).loss
        sliced=answer_loss(model,ids,2)
        self.assertTrue(torch.allclose(full,sliced,atol=1e-6))

    def test_saved_adapter_loads_without_changing_the_base_checkpoint(self):
        import copy
        import torch
        from reader_adapter import attach,load_adapter
        torch.manual_seed(3);model=self.make_model();fresh=copy.deepcopy(model)
        ids=torch.tensor([[1,2,3,4]])
        layers=attach(model,rank=2,alpha=4)
        with torch.no_grad():
            for layer in layers.values():layer.b.fill_(.01)
            expected=model(ids).logits
        weights={name+'.'+part:getattr(layer,part).detach().clone() for name,layer in layers.items() for part in ('a','b')}
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'adapter.pt';torch.save(weights,path)
            load_adapter(fresh,path,dict(rank=2,alpha=4))
        with torch.no_grad():actual=fresh(ids).logits
        self.assertTrue(torch.equal(expected,actual))


if __name__=='__main__':unittest.main()
