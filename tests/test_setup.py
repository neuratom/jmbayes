"""Model set-up: design matrices, separate fits and priors against R."""
import numpy as np
import pytest

from jmbayes import formula as fm
from jmbayes.design import Response, as_terms, ns, var
from jmbayes.fits import fit_coxph
from jmbayes.model import jm_setup

from conftest import README_MODELS, ref, setup_args


def setup(pbc2, pid, **kw):
    return jm_setup(pbc2, pid, *setup_args(), "id", "year", **kw)


def test_design_names_follow_r(pbc2_data):
    pbc2, _ = pbc2_data
    year, sex, drug = var("year"), var("sex"), var("drug")
    assert (year * sex).build(pbc2, {})[1] == ["(Intercept)", "year", "sexfemale",
                                               "year:sexfemale"]
    assert (year + sex + drug).build(pbc2, {})[1] == ["(Intercept)", "year", "sexfemale",
                                                      "drugD-penicil"]
    # R orders main effects before interactions, whatever order they are written in
    assert (year * sex + drug).build(pbc2, {})[1] == ["(Intercept)", "year", "sexfemale",
                                                      "drugD-penicil", "year:sexfemale"]
    assert (year + (year & sex)).build(pbc2, {})[1] == ["(Intercept)", "year",
                                                        "year:sexfemale"]
    assert year.ns(2).build(pbc2, {})[1] == ["(Intercept)", "ns(year, 2)1", "ns(year, 2)2"]


def test_strings_and_objects_give_the_same_terms():
    year, sex, age = var("year"), var("sex"), var("age")
    assert fm.parse_rhs("year * sex + age") == year * sex + age
    assert fm.parse_rhs("year + year:sex") == year + (year & sex)
    assert fm.parse_rhs("ns(year, 2) * sex") == ns("year", 2) * sex
    assert as_terms(["year", ("year", "sex")]) == year + (year & sex)
    assert as_terms("year * sex") == year * sex
    assert fm.parse_response("log(serBilir)") == var("serBilir").log()
    r, idv, unc = fm.parse_random("~ year || id")
    assert r == year and idv == "id" and unc


def test_response_transforms(pbc2_data):
    pbc2, _ = pbc2_data
    assert np.allclose(var("serBilir").log().values(pbc2), np.log(pbc2["serBilir"]))
    a = Response("ascites").values(pbc2.dropna(subset=["ascites"]))
    assert set(np.unique(a)) == {0.0, 1.0}


def test_cox_matches_r(pbc2_data):
    _, pid = pbc2_data
    W = var("sex").without_intercept().build(pid, {})[0]
    beta = fit_coxph(pid["years"].to_numpy(), pid["status2"].to_numpy(), W)
    assert beta[0] == pytest.approx(ref("ref_R_initfits.json")["gammas"], abs=1e-6)


@pytest.fixture(scope="module")
def md_python(pbc2_data):
    return setup(*pbc2_data)


@pytest.fixture(scope="module")
def md_rfits(pbc2_data):
    return setup(*pbc2_data, initial_fits=ref("ref_R_initfits.json"))


def test_linear_mixed_fits_match_r(md_python):
    r = ref("ref_R_initfits.json")
    for j in (0, 1):                                 # lme() fits
        o = md_python.outcomes[j]
        assert md_python.inits["betas"][o.fe_cols] == pytest.approx(r["outcomes"][j]["betas"],
                                                                    rel=1e-4, abs=1e-5)
        D_py = md_python.inits["D"][np.ix_(o.re_cols, o.re_cols)]
        assert np.sqrt(np.diag(D_py)) == pytest.approx(
            np.sqrt(np.diag(np.array(r["outcomes"][j]["D"]))), rel=1e-3)


def test_logistic_mixed_fit_close_to_r(md_python):
    """GLMMadaptive vs our adaptive Gauss-Hermite ML fit: same estimator, ~1% apart."""
    r = ref("ref_R_initfits.json")["outcomes"][2]
    o = md_python.outcomes[2]
    assert md_python.inits["betas"][o.fe_cols] == pytest.approx(r["betas"], rel=0.03)
    D_py = md_python.inits["D"][np.ix_(o.re_cols, o.re_cols)]
    assert np.sqrt(np.diag(D_py)) == pytest.approx(np.sqrt(np.diag(np.array(r["D"]))),
                                                   rel=0.03)


def test_initial_fits_from_r_are_used_exactly(md_rfits):
    r = ref("ref_R_initfits.json")
    sds = np.concatenate([np.sqrt(np.diag(np.array(o["D"]))) for o in r["outcomes"]])
    assert md_rfits.priors["D_sds_mean"] == pytest.approx(sds, rel=1e-12)


def test_priors_follow_jm_rules(md_python):
    pr = md_python.priors
    nb = md_python.W0_H.shape[1]
    assert nb == 11                                   # 9 segments, quadratic B-splines
    assert pr["rank_Tau_bs_gammas"] == nb - 2         # second-order difference penalty
    assert pr["A_tau_bs_gammas"] == 5 and pr["B_tau_bs_gammas"] == 0.5
    assert np.allclose(pr["Tau_alphas"], 0.25 * np.eye(3))
    assert pr["D_sds_shape"] == 5 and pr["D_L_etaLKJ"] == 3
    assert np.all(np.diag(pr["Tau_betas_HC"]) >= 1 / 1000 - 1e-12)


def test_ascites_observation_count(md_python):
    assert len(md_python.outcomes[2].y) == 1885      # as reported by JMbayes2
