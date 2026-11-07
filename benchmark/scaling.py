"""Scaling benchmark: JMbayes2 (R) vs jmbayes (Python, GPU).

Runs each job as its own process, one at a time, and samples the resident memory
of the whole process tree every 0.2 s (R starts one worker process per chain,
so all of them are counted). Python also reports peak GPU memory.

Grid
  patients:   pbc2 copied 1, 2, 5, 10 times (312 .. 3,120 patients), 3,500 iterations
  iterations: 3,500 .. 90,000 iterations at 5 and 10 copies (1,560 and 3,120 patients)
Every job uses 3 chains, burn-in 500, thinning 5 (JMbayes2's defaults except n_iter).
jmbayes runs in float32, its default; earlier float64 results are kept in float64/.

Usage: python scaling.py            (appends to scaling_results.jsonl, skips done jobs)

Each finished job is saved at once. To pause between jobs, create a file named PAUSE in
this folder; delete it and run the script again to continue where it stopped.
"""
import json
import pathlib
import subprocess
import time

import psutil

HERE = pathlib.Path(__file__).resolve().parent
RSCRIPT = r"C:/Program Files/R/R-4.6.2/bin/Rscript.exe"
PY = str(HERE.parent / ".venv" / "Scripts" / "python.exe")
OUT = HERE / "scaling_results.jsonl"
ENV = dict(**__import__("os").environ, R_LIBS_USER="C:/Users/mo/R/win-library/4.6")

GRID = ([(k, 3500) for k in (1, 2, 5, 10)]
        + [(10, n) for n in (7000, 14000, 28000, 60000, 90000)]
        + [(5, n) for n in (7000, 14000, 28000, 60000, 90000)])
R_SKIP = set()               # jobs not run in R (none: R's memory growth is measured)
BURNIN = 500


def tree_rss(p):
    try:
        procs = [p] + p.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    tot = 0
    for q in procs:
        try:
            tot += q.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return tot


def run(cmd, cwd):
    t0 = time.perf_counter()
    p = psutil.Popen(cmd, cwd=cwd, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    peak = 0
    while p.poll() is None:
        peak = max(peak, tree_rss(p))
        time.sleep(0.2)
    out = p.stdout.read().decode(errors="replace")
    if p.returncode:
        raise RuntimeError(out[-2000:])
    return time.perf_counter() - t0, peak / 1e9


def main():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["impl"], r["k"], r["n_iter"]))
    for impl in ("Python-GPU", "R"):              # Python first: its jobs are short
        for k, n_iter in GRID:
            if (impl, k, n_iter) in done or (impl == "R" and (k, n_iter) in R_SKIP):
                continue
            if (HERE / "PAUSE").exists():
                print("paused (delete benchmark/PAUSE and rerun to continue)", flush=True)
                return
            res_file = HERE / f"scal_{impl}_k{k}_it{n_iter}.json"
            if impl == "R":
                cmd = [RSCRIPT, str(HERE / "run_pbc2_R.R"), str(k), str(res_file),
                       str(n_iter), str(BURNIN)]
            else:
                cmd = [PY, str(HERE.parent / "run_pbc2.py"), "--device", "cuda",
                       "--replicate", str(k), "--n-iter", str(n_iter),
                       "--n-burnin", str(BURNIN), "--dtype", "float32",
                       "--out", str(res_file)]
            wall, peak_gb = run(cmd, HERE)
            t = json.loads(res_file.read_text())["timing"]
            rec = dict(impl=impl, k=k, patients=312 * k, n_iter=n_iter, wall_s=wall,
                       peak_ram_gb=peak_gb,
                       setup_s=t.get("separate_fits", t.get("setup")),
                       mcmc_s=t["mcmc"],
                       jm_total_s=t.get("jm_total"),
                       gpu_peak_gb=t.get("gpu_peak_mb", 0) / 1e3)
            with OUT.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{impl:10s} patients={312 * k:5d} iters={n_iter:6d}  wall {wall:7.1f}s  "
                  f"MCMC {t['mcmc']:7.1f}s  peak RAM {peak_gb:5.2f} GB", flush=True)


if __name__ == "__main__":
    main()
