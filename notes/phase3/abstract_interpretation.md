# Transfer from static analysis: know what an abstraction can answer

Cousot and Cousot, **Abstract Interpretation: A Unified Lattice Model for Static
Analysis of Programs by Construction or Approximation of Fixpoints**, POPL 1977,
pp. 238–252, [author-hosted paper](https://www.di.ens.fr/~cousot/publications.www/CousotCousot-POPL-77-ACM-p238--252-1977.pdf).
Targeted reading: introduction, section 6 (p. 242), and interval examples in
section 9. The scanned paper was OCRed locally; section 6 formulas were checked
against the page image. The proceedings cover says Fourth Symposium.

The paper relates abstraction alpha and concretization gamma, requiring concrete
states to be contained in gamma(alpha(states)); compatible abstract operations
preserve that overapproximation. Concretization denotes possible concrete states,
not recovery of an original text. Its fixed-point algorithms concern formal
program semantics, not language-model memory.

Our proposed analogy: a month-valued memory denotes a set of possible event days.
It can support a month query without supporting an exact-day query. An omitted
qualification can similarly prevent a compressed claim from answering a finer
question. A language-model extractor does not automatically satisfy the containment
condition; retaining raw sources elsewhere does not make a false abstraction sound.

The practical research direction is query-specific representation plus source
fallback when an abstraction is insufficient. The current temporal-value routing
is only an empirical heuristic testing that direction. It neither implements
abstract interpretation nor establishes formal soundness. Do not add a lattice,
widening operator or theorem claim merely for mathematical presentation, and do
not launch another experiment before evaluating the frozen routing comparison.
