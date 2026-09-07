import numpy as np

from dm_spikes.capture_fit import fitted_A_B, fitted_capture_factor


PARAMS = {
    "gamma_domain": np.array([0.01, 1.95]),
    "cheb_coeffs_logA": np.array([np.log(1.2), 0.05]),
    "cheb_coeffs_logB": np.array([np.log(2.8), -0.03]),
}


def test_fitted_capture_factor_basic_properties():
    R_S = 1.0
    gamma = np.array([0.01, 1.0, 1.95])
    r = np.array([4.001, 100.0, 1.0e6]) * R_S
    values = fitted_capture_factor(r, gamma, R_S, PARAMS)
    A, B, _ = fitted_A_B(gamma, PARAMS)

    assert np.all(np.isfinite(values))
    assert np.all(A > 0.0)
    assert np.all(B > 0.0)
    assert fitted_capture_factor(4.0 * R_S, 1.0, R_S, PARAMS) == 0.0
    assert fitted_capture_factor(4.0 * R_S * (1.0 + 1e-10), 1.0, R_S, PARAMS) < 1e-20
    assert np.isclose(fitted_capture_factor(1e12 * R_S, 1.0, R_S, PARAMS), fitted_A_B(1.0, PARAMS)[0], rtol=1e-8)


def test_fitted_capture_factor_broadcasting():
    R_S = 1.0
    r = np.array([[5.0], [10.0]]) * R_S
    gamma = np.array([0.01, 1.0, 1.95])
    values = fitted_capture_factor(r, gamma, R_S, PARAMS)
    assert values.shape == (2, 3)
    assert np.all(np.isfinite(values))


if __name__ == "__main__":
    test_fitted_capture_factor_basic_properties()
    test_fitted_capture_factor_broadcasting()
    print("capture_fit basic tests passed")
