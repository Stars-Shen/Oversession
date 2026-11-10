import importlib.util
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires the existing torch environment')
class SourceAttentionTests(unittest.TestCase):
    def model(self):
        from transformers import Qwen3MoeConfig,Qwen3MoeForCausalLM
        cfg=Qwen3MoeConfig(hidden_size=16,intermediate_size=32,moe_intermediate_size=8,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
            num_experts=4,num_experts_per_tok=1,vocab_size=32,sliding_window=None)
        cfg._attn_implementation='eager'
        return Qwen3MoeForCausalLM(cfg).eval()

    def test_attention_row_matches_actual_rope_and_grouped_query_attention(self):
        import torch
        from source_attention import SourceAttention
        torch.manual_seed(1);model=self.model();capture=SourceAttention(model)
        result=model(torch.tensor([[1,2,3,4,5]]),output_attentions=True)
        expected=result.attentions[-1][0,:,3,:4]
        self.assertTrue(torch.allclose(expected,capture.attention(3),atol=1e-6));capture.close()

    def test_source_supervision_cannot_see_future_gold_answer_tokens(self):
        import torch
        from source_attention import SourceAttention,coverage_loss
        from reader_adapter import attach
        torch.manual_seed(2);model=self.model();layers=attach(model,rank=2,alpha=4);capture=SourceAttention(model)
        model(torch.tensor([[1,2,3,4,5]]));first=capture.attention(2)
        model(torch.tensor([[1,2,3,7,8,9]]));second=capture.attention(2)
        self.assertTrue(torch.allclose(first,second,atol=1e-6))
        coverage_loss(second,{'D1:1':(0,1),'D1:2':(1,2)},{'D1:1','D1:2'}).backward()
        self.assertTrue(any(layer.b.grad is not None and layer.b.grad.abs().sum()>0 for layer in layers.values()))
        capture.close()

    def test_joint_sources_need_complementary_head_coverage(self):
        import torch
        from source_attention import coverage_loss
        complementary=torch.tensor([[.8,.1,.1],[.1,.8,.1]])
        repetitive=torch.tensor([[.8,.1,.1],[.8,.1,.1]])
        spans={'D1:1':(0,1),'D1:2':(1,2),'D1:3':(2,3)};positive={'D1:1','D1:2'}
        self.assertLess(coverage_loss(complementary,spans,positive),coverage_loss(repetitive,spans,positive))

    def test_empty_expert_optimization_preserves_input_gradients(self):
        import copy
        import torch
        from local_cad import active_experts_forward
        torch.manual_seed(3);block=self.model().model.layers[0].mlp
        block.requires_grad_(False);other=copy.deepcopy(block)
        x=torch.randn(1,3,16,requires_grad=True);y=x.detach().clone().requires_grad_(True)
        before=block(x)[0];after=active_experts_forward(other,y)[0]
        before.square().sum().backward();after.square().sum().backward()
        self.assertTrue(torch.equal(before,after));self.assertTrue(torch.allclose(x.grad,y.grad,atol=1e-6))


if __name__=='__main__':unittest.main()
