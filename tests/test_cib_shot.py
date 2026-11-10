"""Tests for the CIB Poisson (shot-noise) term, HaloModel.get_power_shot.

Run:  python -m pytest tests/test_cib_shot.py   (or:  python tests/test_cib_shot.py)
"""
import numpy as np

if not hasattr(np, "trapz"):  # hmvec still uses np.trapz (removed in numpy 2)
    np.trapz = np.trapezoid

import hmvec as hm
from hmvec.hmvec import sdndm

AUTO = np.array([[545e9], [545e9]])
PAIR = np.array([[217e9], [545e9]])


def model(nfw=False):
    h = hm.HaloModel(np.geomspace(0.01, 4.0, 15), np.geomspace(1e-3, 10.0, 41), ms=np.geomspace(1e10, 5e15, 30),
                     mass_function="sheth-torman", params={}, mdef="vir", skip_nfw=not nfw)
    h.set_cibParams("planck13", L_o=5.5e-8)
    h.set_cib_subhalo_grid(200)
    return h


def shot(h, nu, **kw):
    return h.get_power_shot("cib", "cib", nu_obs=nu, **kw)[:, 0]


def test_flat_in_k_and_symmetric():
    h = model()
    p = h.get_power_shot("cib", "cib", nu_obs=PAIR)
    assert p.shape == (len(h.zs), len(h.ks)) and np.all(p == p[:, :1])
    assert np.allclose(shot(h, PAIR), shot(h, PAIR[::-1]), rtol=1e-14, atol=0)


def test_no_cut_brute_force():
    """sum over every galaxy of L_a L_b / (4 pi)^2: centrals from get_fcen, satellites explicitly."""
    h = model()
    la, lb = h.cib_lum_z(217e9), h.cib_lum_z(545e9)
    cen = np.trapz(h.nzm * h.cib_sigma_m()[None] ** 2, h.ms, axis=-1)
    satms = h.cib_subhalo_grid()
    sat_m = np.trapz(sdndm(satms, h.ms[:, None], "Tinker") * h.cib_sigma_m(satms) ** 2, satms, axis=-1)
    sat = np.trapz(h.nzm * sat_m[None], h.ms, axis=-1)
    assert np.allclose(shot(h, PAIR), la * lb * (cen + sat), rtol=1e-12, atol=0)
    assert np.allclose(shot(h, PAIR, subhalos=False), la * lb * cen, rtol=1e-12, atol=0)


def test_scatter_without_cut_is_exp_sigma2():
    h = model()
    p0 = shot(h, AUTO)
    # scatter is mean preserving (no effect on 1h/2h) but raises <L^2> = e^{s^2} Lbar^2, with or without a cut
    h.set_cib_fluxcut({}, sigma_ln=0.8)
    assert np.allclose(shot(h, AUTO), np.exp(0.8 ** 2) * p0, rtol=1e-12)
    h.set_cib_fluxcut({545: np.inf}, sigma_ln=0.8)
    assert np.allclose(shot(h, AUTO), np.exp(0.8 ** 2) * p0, rtol=1e-12)


def test_sharp_cut_brute_force_centrals():
    h = model()
    cut = 0.05
    h.set_cib_fluxcut({545: cut})
    s = h.cib_galaxy_flux(545e9)
    assert (s > cut).any()
    cen = np.trapz(h.nzm * (h.cib_sigma_m()[None] ** 2 * (s < cut)), h.ms, axis=-1)
    assert np.allclose(shot(h, AUTO, subhalos=False), h.cib_lum_z(545e9) ** 2 * cen, rtol=1e-12, atol=0)


def test_cut_reduces_and_per_map():
    h = model()
    vals = []
    for cut in (np.inf, 0.3, 0.03, 0.003):
        h.set_cib_fluxcut({545: cut}, sigma_ln=1.0)
        vals.append(shot(h, AUTO).sum())
    assert all(a > b for a, b in zip(vals, vals[1:]))
    h.set_fluxcuts("planck13")
    h.set_cib_fluxcut(per_map=True, sigma_ln=1.0)
    p_map = shot(h, PAIR)
    h.set_cib_fluxcut({217: 0.225, 545: 0.35}, sigma_ln=1.0)
    assert np.allclose(p_map, shot(h, PAIR), rtol=1e-14)


def test_get_power_shot_flag():
    h = model(nfw=True)
    h.set_cib_fluxcut({545: 0.35}, sigma_ln=1.0)
    p = h.get_power("cib", "cib", nu_obs=AUTO)
    ps = h.get_power("cib", "cib", nu_obs=AUTO, shot=True)
    assert np.allclose(ps - p, h.get_power_shot("cib", "cib", nu_obs=AUTO), rtol=1e-12, atol=0)
    try:
        h.get_power_shot("cib", "nfw", nu_obs=AUTO)
    except ValueError:
        pass
    else:
        raise AssertionError("non-CIB shot should raise")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
