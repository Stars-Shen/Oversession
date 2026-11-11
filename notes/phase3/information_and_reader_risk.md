# Information order and a bounded reader

This is a decision-theory interpretation of the completed local experiment,
not another benchmark, implementation claim or reason to launch a parameter grid.

Let H be the complete history including dates/captions, Q the question, A the
target, and R=f(H,Q) the retrieved representation under the fixed writer/index.
For any task loss L and any retrieved-input policy g, a full-information policy
can choose h(H,Q)=g(f(H,Q),Q). Consequently, over unrestricted policies,

    inf_h E[L(h(H,Q), A)] <= inf_g E[L(g(R,Q), A)].

This is a direct feasible-policy argument: the full-information decision maker
can run the existing retriever before deciding. If the writer/retriever is
randomized independently of A conditional on H,Q, allow the full policy the
same independent randomization. No invertibility of a summary or KV cache is
assumed. The comparison concerns optimal attainable risk, not the performance
of a particular fixed neural reader on two different serialized inputs.

The related formalism is Blackwell's comparison of experiments: *Equivalent
Comparisons of Experiments*, Annals of Mathematical Statistics 24(2), 265–272,
1953, [DOI:10.1214/aoms/1177729032](https://doi.org/10.1214/aoms/1177729032).
The primary page is blocked by a service challenge in this session, so no
claim of having read its full proof is made; the elementary argument above is
self-contained. A modern primary abstract also discusses the information-order
equivalence: [Economics Letters, 2024, 112146](https://doi.org/10.1016/j.econlet.2024.112146).
Only its indexed abstract was accessible; no detailed theorem beyond the direct
argument above is imported.

For the actual fixed reader g_theta, there is no such guarantee between
g_theta(H,Q) and g_theta(R,Q): the first evaluation does not execute f internally.
Our full-history **50.97** versus retrieved **58.40** is compatible with this
distinction. It does not demonstrate that the full history has less information,
nor isolate a unique causal explanation such as positional bias. It establishes
that the tested complete-history policy is worse for this reader and metric.

An economics connection is Sims, *Implications of rational inattention*, Journal
of Monetary Economics 50(3), 665–690, 2003,
[DOI:10.1016/S0304-3932(03)00029-1](https://www.sciencedirect.com/science/article/pii/S0304393203000291).
The publisher abstract explicitly models actions through a finite Shannon-capacity
channel. Only the abstract and author-page description were accessible here;
its quantitative macroeconomic model is not transferred. Token count, context
window size and KV bytes are not measured Shannon capacity and cannot be
substituted into that theorem as if they were.

The useful local consequence is narrower: evaluate a memory representation by
the final loss of the bounded reader using it, together with real compute, not
solely by source recall or size. The source-attention adapter, full-history
reader and previous answer-consensus trial all illustrate why an auxiliary
signal is insufficient. This observation does not justify inventing a fitted
mutual-information objective on 80 reused development labels. The next fixed
native 30B policy tests a reader-capability hypothesis with the evidence held
constant; it still needs to beat the strongest cached reader before expansion.

One practical alternative is unavailable offline: the existing SiliconFlow cache
preserves final text and usage but not the provider's reasoning text. Usage
token counts cannot reconstruct teacher trajectories. Do not claim cached
reasoning distillation is possible or launch new paid teacher calls implicitly.
