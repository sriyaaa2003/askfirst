import numpy as np
import pytest

from askfirst.forecast import conformal_quantile
from askfirst.policy import auto_order, cost, human_order, monte_carlo_value, top_fraction


def test_cost_is_zero_when_order_equals_demand():
    assert cost(np.array([5.0]), np.array([5.0]), 0.3, 0.4)[0] == 0.0


def test_cost_weights_shortage_and_surplus():
    assert cost(2.0, 5.0, 0.3, 0.4) == pytest.approx(0.9)   # 3 short at 0.3
    assert cost(5.0, 2.0, 0.3, 0.4) == pytest.approx(1.2)   # 3 over at 0.4


def test_newsvendor_quantile_minimises_expected_cost():
    rng = np.random.default_rng(0)
    y = rng.gamma(4.0, 2.0, 200_000)
    cu, co = 0.3, 0.1
    tau = cu / (cu + co)
    S_star = np.quantile(y, tau)
    best = cost(S_star, y, cu, co).mean()
    for S in (S_star * 0.8, S_star * 1.2):
        assert cost(S, y, cu, co).mean() > best


def test_conformal_quantile_reaches_target_coverage_on_exchangeable_data():
    rng = np.random.default_rng(1)
    calib = rng.standard_normal(5_000)
    test = rng.standard_normal(100_000)
    for tau in (0.1, 0.5, 0.9):
        q = conformal_quantile(calib, tau)
        assert abs((test <= q).mean() - tau) < 0.02


def test_conformal_quantile_never_below_its_finite_sample_rank():
    z = np.arange(1.0, 11.0)
    assert conformal_quantile(z, 0.5) == 6.0     # ceil(11 * 0.5) = 6th smallest
    assert conformal_quantile(z, 0.99) == 10.0   # rank above n: the maximum


def test_planner_with_rho_zero_equals_autonomous_and_rho_one_is_perfect():
    mu, s, q, y = np.array([10.0]), np.array([3.0]), np.array([0.5]), np.array([14.0])
    assert human_order(mu, s, q, y, 0.0) == pytest.approx(auto_order(mu, s, q))
    assert human_order(mu, s, q, y, 1.0) == pytest.approx(y)


def test_value_of_review_is_zero_without_a_planner_and_positive_with_one():
    rng = np.random.default_rng(2)
    n, K = 50, 400
    mu, s = rng.uniform(2, 20, n), rng.uniform(1, 5, n)
    q = np.full(n, 0.3)
    cu, co = np.full(n, 0.3), np.full(n, 0.4)
    z = rng.standard_normal((n, K))
    _, evr0 = monte_carlo_value(mu, s, q, cu, co, z, 0.0)
    risk, evr = monte_carlo_value(mu, s, q, cu, co, z, 0.5)
    assert np.allclose(evr0, 0.0)
    assert (evr > 0).all()
    assert (evr <= risk + 1e-9).all()


def test_uniform_planner_value_is_proportional_to_risk():
    """Documented property of the linear planner: it halves S - y, so EVR = rho * risk."""
    rng = np.random.default_rng(3)
    n, K = 40, 300
    mu, s = rng.uniform(5, 30, n), rng.uniform(1, 4, n)     # mu >> s: no clipping at zero
    q = np.full(n, 0.2)
    cu, co = np.full(n, 0.3), np.full(n, 0.4)
    z = rng.standard_normal((n, K)) * 0.5
    risk, evr = monte_carlo_value(mu, s, q, cu, co, z, 0.5)
    assert np.allclose(evr, 0.5 * risk, rtol=1e-6)


def test_row_specific_planner_changes_the_ranking():
    rng = np.random.default_rng(4)
    n, K = 40, 300
    mu, s = rng.uniform(5, 30, n), rng.uniform(1, 4, n)
    q = np.full(n, 0.2)
    cu, co = np.full(n, 0.3), np.full(n, 0.4)
    z = rng.standard_normal((n, K)) * 0.5
    rho = np.where(np.arange(n) % 2 == 0, 0.9, 0.1)
    risk, evr = monte_carlo_value(mu, s, q, cu, co, z, rho)
    assert np.allclose(evr, rho * risk, rtol=1e-6)
    assert not np.array_equal(np.argsort(-evr), np.argsort(-risk))


@pytest.mark.parametrize("fraction,expected", [(0.0, 0), (0.01, 1), (0.25, 3), (1.0, 10)])
def test_top_fraction_respects_the_budget(fraction, expected):
    score = np.arange(10.0)
    mask = top_fraction(score, fraction)
    assert mask.sum() == (expected if fraction != 0.25 else 3)
    if expected:
        assert score[mask].min() >= np.sort(score)[-expected]
