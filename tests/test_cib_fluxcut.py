"""Tests for the CIB flux cut on the clustered (1h, 2h) power (HaloModel.set_cib_fluxcut).

Run:  python -m pytest tests/test_cib_fluxcut.py   (or:  python tests/test_cib_fluxcut.py)
"""
import numpy as np

if not hasattr(np, "trapz"):  # hmvec still uses np.trapz (removed in numpy 2)
    np.trapz = np.trapezoid

import hmvec as hm

PAIR = np.array([[217e9], [545e9]])
AUTO = np.array([[545e9], [545e9]])


def model():
    h = hm.HaloModel(np.linspace(0.02, 4.0, 12), np.geomspace(1e-3, 10.0, 61), ms=np.geomspace(1e10, 5e15, 25),
                     mass_function="sheth-torman", params={}, mdef="vir")
    h.set_cibParams("planck13", L_o=5.5e-8)
    return h


def power(h, nu):
    return (h.get_power_1halo("cib", "cib", nu_obs=nu), h.get_power_2halo("cib", "cib", nu_obs=nu))


def test_no_cut_is_default():
    h = model()
    p0 = power(h, PAIR)
    h.set_cib_fluxcut(None)
    assert all(np.array_equal(a, b) for a, b in zip(p0, power(h, PAIR)))
    h.set_cib_fluxcut({}, sigma_ln=0.7)     # scatter alone (no cut) changes nothing
    assert all(np.array_equal(a, b) for a, b in zip(p0, power(h, PAIR)))


def test_infinite_cut_reproduces_uncut():
    h = model()
    p0 = power(h, PAIR)
    for s in (0.0, 1.0):
        h.set_cib_fluxcut({545: np.inf}, sigma_ln=s)
        for a, b in zip(p0, power(h, PAIR)):
            assert np.allclose(a, b, rtol=1e-13, atol=0)


def test_sharp_cut_centrals_brute_force():
    h = model()
    cut = 0.05  # Jy at 545 GHz: removes bright low-z centrals on this grid
    s = h.cib_galaxy_flux(545e9)
    assert (s > cut).any() and (s < cut).any()
    ref = h.get_fcen(np.array([545e9]))[..., 0] * (s < cut)
    got = h.get_fcen_cut(np.array([545e9]), {545: cut})[..., 0]
    assert np.allclose(got, ref, rtol=1e-13, atol=0)


def test_deeper_cut_removes_more():
    h = model()
    vals = []
    for cut in (np.inf, 1.0, 0.1, 0.01):
        h.set_cib_fluxcut({545: cut}, sigma_ln=0.5)
        p1, p2 = power(h, AUTO)
        vals.append(p1.sum() + p2.sum())
    assert all(a >= b for a, b in zip(vals, vals[1:])) and vals[-1] < vals[0]


def test_per_map_uses_flux_cuts_and_pair_bands():
    h = model()
    h.set_fluxcuts("planck13")
    h.set_cib_fluxcut(per_map=True, sigma_ln=1.0)
    assert h._cib_pair_cuts(PAIR) == {217: 0.225, 545: 0.35}
    assert h._cib_pair_cuts(AUTO) == {545: 0.35}
    # per-map auto == common mask with that band's cut alone
    p_map = power(h, AUTO)
    h.set_cib_fluxcut({545: 0.35}, sigma_ln=1.0)
    assert all(np.allclose(a, b, rtol=1e-13, atol=0) for a, b in zip(p_map, power(h, AUTO)))
    h.set_cib_fluxcut({100: 0.4}, per_map=True)
    try:
        power(h, AUTO)
    except ValueError:
        pass
    else:
        raise AssertionError("missing per-map cut should raise")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
