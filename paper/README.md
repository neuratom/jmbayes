# Paper

`main.tex` is the paper; `main.pdf` is the compiled version. To rebuild:
`pdflatex main.tex` twice.

## Submitting to arXiv

1. Upload `arxiv_submission.zip` (main.tex, the two table files and the two figures). arXiv compiles the
   LaTeX itself with pdflatex. To recreate the zip after editing, include `main.tex`,
   `ext_table.tex`, `hr_table.tex` and `figures/*.png`, nothing else (the tables are
   written by `../benchmark/hr_table.py`).
2. Fill in the form:
   * **Title:** jmbayes: fast, memory-efficient, data-parallel Bayesian joint models for
     longitudinal and time-to-event data
   * **Authors:** M. Assaf
   * **Abstract:** copy it from `main.tex`, without the LaTeX commands (it is about 1,900
     characters; arXiv's limit is 1,920)
   * **Primary category:** stat.CO (Statistics - Computation)
   * **Cross-lists:** stat.ME (Methodology), cs.MS (Mathematical Software)
   * **Comments:** 12 pages, 2 figures, 4 tables. Code: https://github.com/neuratom/jmbayes
   * **Licence:** CC BY 4.0 is the usual choice for open work
3. A first submission to a category may need an endorsement from someone who has already
   published there; arXiv tells you after you submit if you need one.

Before submitting, update the benchmark table if more JMbayes2 runs have finished
(`../benchmark/scaling_results.jsonl`) and copy the latest `benchmarks.png` from
`../benchmark/figures/`.
