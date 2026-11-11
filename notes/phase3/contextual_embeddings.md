# Context before pooling: a representation transfer to assess

Read while the independent turn-writer build runs; this is not another launched
experiment or authorization to change its frozen input policy.

Günther, Mohr, Wang and Xiao, **Late Chunking: Contextual Chunk Embeddings Using
Long-Context Embedding Models**, [arXiv:2409.04701v1](https://arxiv.org/html/2409.04701v1),
7 September 2024. Read the full short v1, including method, both result tables,
and conclusion/limitations. The authors' code is linked in the paper; no code
was copied, model downloaded, or environment changed for this reading.

The idea addresses a relevant representation failure: independently embedding a
sentence such as "the city" or "its inhabitants" removes the antecedent. Encode
the long document first, then mean-pool token vectors within each chunk. This
lets a chunk's vector retain context outside its span without displaying the
whole document to the final reader. Their v1 retrieval test uses one model,
jina-embeddings-v2-small, with 256-token chunks on five BEIR datasets. SciFact
nDCG@10 rises 64.20 -> 66.10; NFCorpus rises 23.46 -> 29.98, while full-document
encoding reaches 30.40. Quora is unchanged. These are document-retrieval results,
not dialogue QA or our official LoCoMo F1; the paper explicitly notes its narrow
evaluation setting. Its broad applicability statement is not a matched test on
the current encoder.

The local Qwen3-Embedding-0.6B checkpoint's `1_Pooling/config.json` enables
`pooling_mode_lasttoken`, not mean pooling. Blindly replacing this trained
pooling rule by the paper's mean pooling would change the representation in an
unsupported way. Also distinguish full-document contextualization from a causal
writer: bidirectional chunk vectors can encode future turns. A causal adaptation
would need prefix-only context, preserved target boundaries and original
last-token pooling. It might preserve names/relations before retrieval without
LLM-generated summary text, but could also diffuse the target into the surrounding
topic. It is a hypothesis, not an implication of the BEIR gains or a sufficiency
guarantee.

If this direction is tested later, preserve raw evidence, the final 3K budget,
Luna reader, and source-RRF packing. Change a single representation policy,
check causal boundaries once, and decide by final QA rather than embedding
similarity illustrations. Do not start a context-length, pooling, model, or
chunk-size grid alongside the current writer comparison.
