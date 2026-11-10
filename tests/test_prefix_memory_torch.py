import importlib.util
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires the existing torch environment')
class PrefixMemoryTests(unittest.TestCase):
    def test_two_independent_answers_match_fresh_full_context_and_preserve_memory(self):
        import torch
        from transformers import Qwen3MoeConfig,Qwen3MoeForCausalLM,GenerationConfig
        from prefix_memory import prefill,answer_from_prefix
        torch.manual_seed(20261111)
        config=Qwen3MoeConfig(hidden_size=16,intermediate_size=32,moe_intermediate_size=8,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
            num_experts=4,num_experts_per_tok=1,vocab_size=32,sliding_window=None,
            bos_token_id=1,eos_token_id=31,pad_token_id=0)
        config._attn_implementation='sdpa';model=Qwen3MoeForCausalLM(config).eval()
        prefix=torch.tensor([[3,4,5,6,7]])
        questions=[torch.tensor([[3,4,5,6,7,8,9]]),torch.tensor([[3,4,5,6,7,10,11,12]])]
        gen=GenerationConfig(do_sample=False,max_new_tokens=5,eos_token_id=31,pad_token_id=0)
        cache=prefill(model,prefix);original=[x.clone() for x in cache.key_cache]
        for ids in questions:
            with torch.inference_mode():
                expected=model.generate(ids,attention_mask=torch.ones_like(ids),generation_config=gen,
                    do_sample=False,temperature=1.,top_p=1.,top_k=50,logits_to_keep=1)
            actual,cache=answer_from_prefix(model,ids,cache,prefix.shape[1],gen)
            self.assertTrue(torch.equal(expected,actual))
            self.assertEqual(cache.get_seq_length(),prefix.shape[1])
            for before,after in zip(original,cache.key_cache):self.assertTrue(torch.equal(before,after))


if __name__=='__main__':unittest.main()
