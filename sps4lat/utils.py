r"""
Useful functions for semi-parametric component separation.
This module implements routines that will be used in the rest of the project.
"""

import numpy as np
import healpy as hp
import sys

__all__ = ['sort_alms_ell', 'get_alms']


def sort_alms_ell(alms):
    """ Function to sort alms by ell.

    Healpy ordering is (ell,m) = ((0,0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2))
    New ordering will be (ell,m) = ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)).
    This enables faster evaluation of the empirical covariance matrix.
    Parameters
    ----------
    alms : ndarray
        alms to sort by ell, stored in an array of (N_freqs * size)
    Returns
    -------
    alms_sorted : ndarray
        array of alms sorted by ell

    """
    if alms.ndim == 1:
        alms = [alms]
    size = len(alms[0])
    lmax = hp.Alm.getlmax(size)
    idx_sort = [0]
    for ll in range(1, lmax + 1):
        idx_sort += [ll + i * lmax - int((i - 1) * i * .5) for i in
                     range(ll + 1)]
    alms_sorted = []
    for alm in alms:
        alms_sorted.append(alm[idx_sort])
    alms_sorted = np.array(alms_sorted)
    return alms_sorted


def get_alms(maps, beams=None, lmax=None):
    """ Function to get alms from maps.

    Correct for the beams (if any). Beams are assumed to be gaussian.
    Parameters
    ----------
    maps : list
        list containing the frequency maps that can have different nside
    beams : ndarray
        beams associated with each map (assumed gaussian)
    lmax : int
        max ell to compute the alms
    Returns
    -------
    alms : ndarray
        array storing the shperical harmonic transform od each freq map.

    """
    if not isinstance(maps, list):
        maps = [maps]
    lmax_maps_list = [3 * hp.get_nside(fmap) - 1 for fmap in maps]
    if lmax is None:
        lmax = min(lmax_maps_list)
    else:
        try:
            assert (all([lmax <= ll for ll in lmax_maps_list]))
        except AssertionError:
            sys.exit("Some maps do not have the neccesary resolution to "
                     "resolve lmax={:d}, consider using a smaller "
                     "lmax".format(int(lmax)))
    alms = []
    for f, fmaps in enumerate(maps):
        alms.append(hp.map2alm(fmaps, lmax=lmax))
    alms = np.array(alms)
    if beams is not None:
        for fwhm, alm in zip(beams, alms):
            lmax = hp.Alm.getlmax(len(alm))
            bl = hp.gauss_beam(np.radians(fwhm / 60.0), lmax)
            for i_alm, i_bl in zip(alm, bl.T):
                hp.almxfl(i_alm, 1.0 / i_bl, inplace=True)
    return alms


if __name__ == '__main__':
    nside = 2
    npix = hp.nside2npix(nside)
    map_test = np.random.randn(npix)
    alm_test = get_alms(map_test, beams=np.array([1.]))
    alm_sorted = sort_alms_ell(alm_test)
    idx = [np.argwhere(alm_test == alm) for alm in alm_sorted]
    print(idx)
