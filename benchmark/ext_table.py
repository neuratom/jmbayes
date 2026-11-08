"""Markdown table of the extension comparisons (README), from <model>_equivalence.json
written by compare_cr.py."""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROWS = [
    ("surv", "Strata (drug) + late entry; log bilirubin and prothrombin", "12 x 30,000"),
    ("fam", "Negative binomial, Poisson and Student's t (df 4) markers", "12 x 30,000"),
    ("tvc", "Time-varying covariate (edema), counting-process rows", "8 x 15,000"),
    ("icens", "Interval-censored event times", "8 x 15,000"),
    ("crx", "Competing risks + late entry + strata (drug)", "8 x 15,000"),
    ("bystr", "Strata with effects per stratum; 2 markers", "8 x 15,000"),
    ("rec", "Recurrent + terminal events, frailty, gap timescale (vignette data)",
     "8 x 15,000"),
    ("fam2", "Gamma, beta and censored normal markers", "8 x 15,000"),
    ("fam3", "Binomial with trials and beta-binomial markers", "8 x 15,000"),
    ("f8a", "Value + velocity (backward) + area over 2 years; spline marker", "8 x 15,000"),
    ("f8b", "expit(value) + Dexp(slope) + acceleration; spline marker", "8 x 15,000"),
    ("f8c", "Value by sex + Delta over 3 years; spline marker", "8 x 15,000"),
    ("tva", "Time-varying association: value x ns(year, 3 knots)", "8 x 15,000"),
    ("bhw", "Weibull baseline hazard", "8 x 15,000"),
    ("bhp", "Piecewise-constant baseline hazard", "8 x 15,000"),
    ("bhn", "Natural cubic spline baseline hazard", "8 x 15,000"),
    ("ms", "Multi-state (illness-death), Weibull hazards (vignette data)", "8 x 15,000"),
    ("wts", "Case weights", "8 x 15,000"),
    ("pen", "Horseshoe prior on associations, ridge on covariates; 2 markers",
     "8 x 15,000"),
    ("aids", "AIDS data (467 patients): square-root CD4, drug + previous infection",
     "8 x 15,000"),
    ("prothro", "Prothrombin data (488 cirrhosis patients): prednisone vs placebo",
     "8 x 15,000"),
]


def fmt_t(sec):
    return f"{sec / 60:.1f} min" if sec >= 60 else f"{sec:.0f} s"


def main():
    out = ["| Model | Chains x iterations | Equivalent (±0.1 SD / ±0.2 SD) | Shown to differ "
           "| Largest difference |",
           "|---|---|---|---|---|"]
    for key, label, size in ROWS:
        f = HERE / f"{key}_equivalence.json"
        if not f.exists():
            continue
        e = json.loads(f.read_text())
        n = e["n_parameters"]
        out.append(f"| {label} | {size} | {e['equivalent_0_1']} / {e['equivalent_0_2']} of "
                   f"{n} | {e['shown_to_differ']} | {e['max_abs_diff_sd']:.3f} SD |")
    print("\n".join(out))
    return "\n".join(out)


if __name__ == "__main__":
    main()
