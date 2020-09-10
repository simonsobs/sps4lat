""" module preprocessing
This module implements pre-processing of maps before computation of
the empirical covariance matrix. Computing alms from maps, deconvolving beam,
discarding ells<ell_min and ells>ell_max, correcting for eventual mask.
"""
import numpy as np
import healpy as hp


def alms_from_maps(maps, beams=None):
    """ Function to get alms from healpix maps.

    Correct for the beams (if any). Beams are assumed to be gaussian.
    Parameters
    ----------
    maps : list
        list containing the frequency maps that can have different nside
    beams : ndarray
        beams associated with each map (assumed gaussian)
    Returns
    -------
    alms : ndarray
        array storing the spherical harmonic transform of each freq map.
        Shape is ``(freqs, lm, ri)``

    """
    if not isinstance(maps, list):
        maps = [maps]
    lmax_maps_list = [3 * hp.get_nside(fmap) - 1 for fmap in maps]
    lmax = min(lmax_maps_list)
    alms = []
    for f, fmaps in enumerate(maps):
        alms.append(hp.map2alm(fmaps, lmax=lmax))
    if beams is not None:
        for fwhm, alm in zip(beams, alms):
            lmax = hp.Alm.getlmax(len(alm))
            bl = hp.gauss_beam(np.radians(fwhm / 60.0), lmax)
            for i_alm, i_bl in zip(alm, bl.T):
                hp.almxfl(i_alm, 1.0 / i_bl, inplace=True)
    return np.array(alms)


def empirical_covmat(alms, lmin=None, lmax=None):
    """ Computes the empirical covariance matrix from alms.

    Parameters
    ----------
    alms : ndarray
        array storing the spherical harmonic transform of each freq map.
        Shape is ``([Stokes], freqs, lm)``
    lmin : int
        min ell to include in the covariance matrix
    lmax : int
        max ell to include in the covariance matrix

    Returns
    -------
    res : ndarray
        shape is ``(...,freqs, freqs,ell)``
    """

    if not lmax:
        lmax = hp.Alm.getlmax(alms.shape[-1])
    if not lmin:
        lmin = 2
    alms = np.array(alms, copy=False, order='C')
    alms = alms.view(np.float64).reshape(alms.shape + (2,))
    if alms.ndim > 3:  # Shape has to be ([Stokes], freq, lm, ri)
        alms = alms.transpose(1, 0, 2, 3)

    res = (alms[..., np.newaxis, :, :lmax + 1, 0]
           * alms[..., :, np.newaxis, :lmax + 1, 0])  # (Stokes, freq,
    # freq, ell)

    consumed = lmax + 1
    for i in range(1, lmax + 1):
        n_m = lmax + 1 - i
        alms_m = alms[..., consumed:consumed + n_m, :]
        res[..., i:] += 2 * np.einsum('...fli,...nli->...fnl', alms_m,
                                      alms_m)
        consumed += n_m

    res /= 2 * np.arange(lmax + 1) + 1
    return res
