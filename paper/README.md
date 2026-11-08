# Paper

`main.tex` is the paper; `main.pdf` is the compiled version. To rebuild:
`pdflatex main.tex` twice. The tables are written by `../benchmark/hr_table.py` (and
`multi_compare.py`), the hazard-ratio figures by `../benchmark/hr_figures.py`.

## Submitting to arXiv

Before you start:

* **Make the GitHub repository public.** The abstract links to
  https://github.com/neuratom/jmbayes, and arXiv requires links to code to resolve
  publicly. (On 2026-11-09 the link gave 404 to a visitor who is not signed in.)
* **Endorsement.** A first submission to a category needs an endorsement. Since January
  2026 the automatic route needs both an academic email address and a prior paper in the
  field, so expect to need a personal endorsement: start the submission, choose stat.CO,
  and arXiv emails you a link with a six-character code. Send it to someone who has
  published in stat.CO and knows the work (the "Which authors of this paper are
  endorsers?" link on arXiv abstract pages shows who can endorse). Endorsement is not peer
  review, and arXiv staff cannot give one.
* arXiv now allows two submissions per month per author.

Then:

1. Run `python make_arxiv.py`. It writes `arxiv_submission.zip` (main.tex, the table files
   and the figures it uses, nothing else) and `arxiv_abstract.txt` (the abstract in plain
   ASCII, checked against arXiv's 1,920-character limit). Upload the zip and choose
   pdflatex; arXiv compiles it itself.
2. Fill in the form:
   * **Title:** jmbayes: fast, memory-efficient, data-parallel Bayesian joint models for
     longitudinal and time-to-event data
   * **Authors:** M. Assaf (Neuratom)
   * **Abstract:** paste `arxiv_abstract.txt` (the form accepts ASCII only)
   * **Primary category:** stat.CO (Statistics - Computation)
   * **Cross-lists:** stat.ME (Methodology), cs.MS (Mathematical Software)
   * **Comments:** the page, figure and table counts of the final PDF, then
     "Code: https://github.com/neuratom/jmbayes"
   * **Licence:** CC BY 4.0 is the usual choice for open work (the choice cannot be
     changed later)
3. Check the PDF arXiv builds before the final step; until it is announced you can still
   withdraw it with "Unsubmit" and fix it.

Checked against arXiv's rules (2026-11-09): pdflatex with PNG figures in a subfolder,
inline bibliography, 11 pt type, 1-inch margins, file names arXiv accepts, no build files
in the zip, the use of an LLM disclosed in the methods and not listed as an author, the
zip compiles from scratch in an empty folder.

Before submitting, update the benchmark table if more JMbayes2 runs have finished
(`../benchmark/scaling_results.jsonl`) and copy the latest `benchmarks.png` from
`../benchmark/figures/`.
