"""Build the arXiv upload: arxiv_submission.zip (main.tex, the \\input table files and the
figures, nothing else) and arxiv_abstract.txt (the abstract as plain ASCII, ready to paste
into arXiv's form, which accepts ASCII only and at most 1,920 characters).

Usage (from paper/): python make_arxiv.py
"""
import glob
import re
import zipfile

s = open("main.tex", encoding="utf-8").read()
ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S).group(1)
t = " ".join(ab.split())
for a, b in [(r"\textbf{jmbayes}", "jmbayes"), (r"$\times$", "x"), ("~", " "),
             ("--", "-"), (r"\texttt{pip install jmbayes}", "pip install jmbayes")]:
    t = t.replace(a, b)
t = re.sub(r"\\url\{([^}]*)\}", r"\1", t)
assert all(ord(c) < 128 for c in t), "non-ASCII character in the abstract"
assert "\\" not in t and "$" not in t, "LaTeX left in the abstract: " + t
assert len(t) <= 1920, f"abstract too long: {len(t)} characters"
open("arxiv_abstract.txt", "w", encoding="ascii").write(t + "\n")

inputs = re.findall(r"\\input\{([^}]*)\}", s)
figs = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", s)
files = ["main.tex", *inputs, *figs]
for f in files:
    assert re.fullmatch(r"[A-Za-z0-9_+\-.,=/]+", f), f"name arXiv rejects: {f}"
unused = set(p.replace("\\", "/") for p in glob.glob("figures/*")) - set(figs)
with zipfile.ZipFile("arxiv_submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f)
print(f"abstract {len(t)} characters; zip: {files}")
if unused:
    print("not included (unused):", sorted(unused))
