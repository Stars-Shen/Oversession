"""Run with the existing torch environment; no model downloads or paid calls."""
import importlib.util
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires the existing torch environment')
class LocalDecodingTests(unittest.TestCase):
    def test_skipping_empty_experts_preserves_routing_math(self):
        import torch
        from transformers import Qwen3MoeConfig
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
        from local_cad import active_experts_forward
        torch.manual_seed(0)
        cfg=Qwen3MoeConfig(hidden_size=16,moe_intermediate_size=8,num_experts=8,num_experts_per_tok=2)
        block=Qwen3MoeSparseMoeBlock(cfg).eval()
        for shape in [(1,1,16),(2,4,16),(1,16,16)]:
            x=torch.randn(*shape)
            with torch.inference_mode():old=block(x);new=active_experts_forward(block,x)
            self.assertTrue(torch.equal(old[0],new[0]))
            self.assertTrue(torch.equal(old[1],new[1]))

    def test_zero_alpha_and_greedy_override_reproduce_original_tokens(self):
        import torch
        from transformers import Qwen3MoeConfig,Qwen3MoeForCausalLM,GenerationConfig
        cfg=Qwen3MoeConfig(hidden_size=16,intermediate_size=32,moe_intermediate_size=8,
                          num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
                          head_dim=8,num_experts=4,num_experts_per_tok=2,vocab_size=32,
                          bos_token_id=0,eos_token_id=None,pad_token_id=0)
        model=Qwen3MoeForCausalLM(cfg).eval();model.generation_config.do_sample=True
        captured=[];original=model._sample
        def tracked(*args,**kwargs):
            captured.append(kwargs['generation_config'].do_sample)
            return original(*args,**kwargs)
        model._sample=tracked
        ids=torch.tensor([[1,2,3]])
        options=dict(attention_mask=torch.ones_like(ids),do_sample=False,
                     generation_config=GenerationConfig(do_sample=False,max_new_tokens=3,pad_token_id=0,bos_token_id=0),
                     temperature=1.0,top_p=1.0,top_k=50)
        torch.manual_seed(1);plain=model.generate(ids,**options)
        torch.manual_seed(99);zero_alpha=model.generate(ids,guidance_scale=1.0,negative_prompt_ids=torch.tensor([[3,4]]),**options)
        self.assertTrue(torch.equal(plain,zero_alpha))
        self.assertEqual(captured,[False,False])


if __name__=='__main__':unittest.main()
