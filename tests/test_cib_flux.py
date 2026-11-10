"""Tests for the per-galaxy CIB flux helpers and the sub-halo grid option.

Run:  python -m pytest tests/test_cib_flux.py   (or:  python tests/test_cib_flux.py)
"""
import numpy as np

if not hasattr(np, "trapz"):  # hmvec still uses np.trapz (removed in numpy 2)
    np.trapz = np.trapezoid

import hmvec as hm
from hmvec.hmvec import sdndm

NU = 545e9


def model():
    h = hm.HaloModel(np.linspace(0.1, 4.0, 10), np.geomspace(1e-3, 10.0, 51), ms=np.geomspace(1e10, 5e15, 20),
                     mass_function="sheth-torman", params={}, mdef="vir", skip_nfw=True)
    h.set_cibParams("planck13", L_o=5.5e-8)
    return h


def test_central_flux_matches_get_fcen():
    h = model()
    chi = h.comoving_radial_distance(h.zs)
    ref = h.get_fcen(np.array([NU]))[..., 0] / (chi[:, None] ** 2 * (1 + h.zs[:, None]))
    assert np.allclose(h.cib_galaxy_flux(NU), ref, rtol=1e-13, atol=0)


def test_lum_factorisation():
    h = model()
    lum = h.cib_lum_z(NU)[:, None] * h.cib_sigma_m()[None, :]
    assert np.allclose(lum, h.get_fcen(np.array([NU]))[..., 0], rtol=1e-13, atol=0)


def test_get_flux_is_central_plus_summed_satellites():
    h = model()
    satms = h.cib_subhalo_grid()
    sat = np.trapz(sdndm(satms, h.ms[:, None], "Tinker") * h.cib_galaxy_flux(NU, satms), satms, axis=-1)
    ref = h.get_flux(NU)[..., 0]
    assert np.allclose(h.cib_galaxy_flux(NU) + sat, ref, rtol=1e-12, atol=0)


def test_galaxy_flux_shapes_and_arbitrary_z():
    h = model()
    assert h.cib_galaxy_flux(NU, ms=1e12, zs=[0.01, 0.5]).shape == (2,)
    assert h.cib_galaxy_flux(NU, ms=np.ones((3, 4)) * 1e12).shape == (len(h.zs), 3, 4)
    # nearer galaxies are brighter
    s = h.cib_galaxy_flux(NU, ms=10 ** 12.6, zs=[0.01, 0.1, 1.0])
    assert s[0] > s[1] > s[2]


def test_subhalo_grid_default_and_convergence():
    h = model()
    assert h.cib_subhalo_grid().shape == (len(h.ms), len(h.ms) + 1)
    fsat_default = h.get_fsat(np.array([NU]))[..., 0]
    h.set_cib_subhalo_grid(400)
    fsat_400 = h.get_fsat(np.array([NU]))[..., 0]
    h.set_cib_subhalo_grid(2000)
    fsat_2000 = h.get_fsat(np.array([NU]))[..., 0]
    big = h.ms > 1e11
    assert np.max(np.abs(fsat_400[:, big] / fsat_2000[:, big] - 1)) < 1e-3
    excess = fsat_default[:, big] / fsat_2000[:, big] - 1   # coarse default overestimates, by < ~7%
    assert excess.min() > 0 and excess.max() < 0.08
    h.set_cib_subhalo_grid(None)
    assert np.array_equal(h.get_fsat(np.array([NU]))[..., 0], fsat_default)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
