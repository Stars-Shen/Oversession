# Category mixture changes the practical meaning of a gain

`scripts/audit_category_mixture.py` uses only saved, already scored predictions.
It preserves measured cohort scores and computes the sensitivity formula
`delta_mix = sum_c(target_count_c / target_n * observed_mean_delta_c)`.
This is conditional on within-category effects transferring. It is not a
completed benchmark score; interrupted generation can select nonrandom questions
even within a category.

Full LoCoMo categories 1–4 contain 282 multi-hop, 321 temporal, 96 inference and
841 single-hop questions. Thus single-hop constitutes **54.61%**. The 160-question
validation cohort has 40 of each category. Its observed DeepSeek thinking gain
is +5.7172 points; weighting its category effects like the 1307-question separate-
conversation pool gives +4.3699 as a sensitivity calculation.

The more consequential issue concerns the partial confirmation cohort:

| Category | Completed confirmation pairs | Eligible confirmation pool |
|---|---:|---:|
| Multi-hop | 176 | 199 |
| Temporal | 174 | 218 |
| Inference | 43 | 43 |
| Single-hop | 210 | 687 |

The pool excludes all 233 questions in development conversations and the 160
validation questions, leaving 1147. The measured 603-pair gain **+3.9006** is
correct for that cohort. However, thinking's single-hop category effect there is
−4.0192 points. Applying the eligible pool's category proportions reduces the
conditional sensitivity gain to **+0.9326**. Applying full-benchmark category
proportions instead gives +1.6289. Neither is an observed final score or a
replacement for the missing answers.

Decision: do not promote a balanced-sample or interrupted-cohort gain into a
full-micro-F1 breakthrough. Preserve raw cohort metrics, category effects and
completion scope. Native local reasoning remains a new experiment; it must show
a real absolute gain before expansion, with attention to the dominant single-hop
task. Do not tune a category oracle or change the official scorer to raise scores.
