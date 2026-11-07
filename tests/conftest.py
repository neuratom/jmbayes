import json
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent


def ref(name):
    """R reference results (JMbayes2 0.6-5), produced by benchmark/run_pbc2_R.R and
    benchmark/export_initfits.R."""
    return json.loads((HERE / name).read_text())


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow MCMC tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long MCMC run, needs --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def pbc2_data():
    import jmbayes as jmb
    return jmb.datasets.pbc2()


README_MODELS = [
    dict(formula="log(serBilir) ~ year * sex", random="~ year | id"),
    dict(formula="prothrombin ~ year * sex", random="~ year | id"),
    dict(formula="ascites ~ year + sex", random="~ year | id", family="binomial"),
]


def setup_args():
    """README models as jm_setup() input (long_models, surv), built via the public API."""
    import jmbayes as jmb
    mk = [jmb.Longitudinal(**m).to_marker() for m in README_MODELS]
    long_models = [dict(response=m.response, fixed=m.fixed, random=m.random, family=m.family)
                   for m in mk]
    ev = jmb.Event("years", "status2", "sex")
    return long_models, dict(time=ev.time, event=ev.event, covariates=ev.covariates)
