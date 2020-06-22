""" module covariance
This module implements computation of the empirical covariance matrix.
"""
import numpy as np
from sps4lat import utils as utl
import os
import healpy as hp


class Experiment:
    def __init__(self, name, lmin, lmax, freqs, bin_size=None, beams=None):
        self.name = name
        self.lmax = lmax
        self.lmin = lmin
        self.beams = beams
        self.freqs = freqs
        self.bin_size = bin_size
        ells = np.linspace(0, lmax, lmax + 1)
        if bin_size is None:
            self.ell_mean = ells
        else:
            self.ell_mean = utl.get_ell_mean(lmin, lmax, bin_size)
        self.maps = None

    def read_map_file(self, file):
        """
        Read maps from fits file.
        Parameters
        ----------
        file : str
            root of the files containing maps. Each file is located at
            file + "{:freq}GHz.fits"

        """
        file_list = [os.path.join(file, "{:d}GHZ.fits".format(int(fr))) for
                     fr in self.freqs]
        self.maps = [hp.read_map(f, verbose=False) for f in file_list]

    @property
    def empirical_covmat(self):
        """ """
        if hasattr(self, 'emp_cov'):
            return self.emp_cov
        else:
            cov = _get_covmat_maps(self.maps, lmax=self.lmax,
                                   beams=self.beams)
            if self.bin_size is None:
                res_binned = cov
            else:
                res_binned = utl.bin_spectrum(cov, self.lmin, self.lmax,
                                              self.bin_size)
            self.emp_cov = res_binned
            return res_binned


def _get_covmat_maps(map_list, lmax=None, beams=None):
    """
     Compute empirical covariance matrix from list of maps.
    Parameters
    ----------
    map_list : list
        list of maps, possibly with different nside
    lmax : int
        ell max to include, if None, takes the ell max permitted by the
        lowest resolution map
    beams : ndarray
        beams to deconvolute to the maps
    Returns
    -------
    res : ndarray
        shape is ``(...,freqs,freqs,ell)``
    """
    alms = utl.get_alms(map_list, lmax=lmax, beams=beams)
    res = _empirical_harmonic_covariance(alms)
    return res


def _empirical_harmonic_covariance(alms):
    """
    Given a list of alm, efficiently computes the empirical covariance matrix
    Parameters
    ----------
    alms : ndarray
        shape is ``([stokes], freqs, lm)``
    Returns
    -------
    res : ndarray
        shape is ``(...,freqs, freqs,ell)``
    """
    alms = np.array(alms, copy=False, order='C')
    alms = alms.view(np.float64).reshape(alms.shape + (2,))
    if alms.ndim > 3:  # Shape has to be ([Stokes], freq, lm, ri)
        alms = alms.transpose(1, 0, 2, 3)
    lmax = hp.Alm.getlmax(alms.shape[-2])

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


def _get_covmat_subdomain(alms_sorted, lmin, lmax):
    """
    Compute the empirical covariance matrix between lmin and lmax.
    Parameters
    ----------
    alms_sorted : ndarray
        array containing the alms at each frequency : shape is
        ``(freqs,lm,ri)``
    lmin : int
        smallest ell of the subdomain (included)
    lmax : int
        largest ell of the subdomain (included)
    Returns
    -------
    covmat : ndarray
        empirical covariance over the sudomain : shape is ``(freqs,freqs)``

    """
    idx_start, idx_end = _get_subdomain(lmin, lmax)
    n_p = idx_end - idx_start
    covmat = 1. / n_p * np.einsum('is,js ->ij',
                                  alms_sorted[:, idx_start:idx_end],
                                  alms_sorted[:, idx_start:idx_end].conj())
    return covmat.real.astype(np.float64)


def _get_subdomain(lmin, lmax):
    """ Returns start and end indices.

    Assuming alms are sorted by ell returns indices such that
    alms[idx_start:idx_end] corresponds to subdomain from lmin to lmax
    Parameters
    ----------
    lmin : int
        lmin of the subdomain (inclusive)
    lmax : int
        lmax of the subdomain (inclusive)

    Returns
    -------
    idx_start :int
    idx_end :int
    """
    idx_start = int(lmin * (lmin + 1) * .5)
    idx_end = int((lmax + 1) * (lmax + 2) * .5)
    return idx_start, idx_end


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    freqs = np.array([50., 100., 150., 200.])
    N = len(freqs)
    nside = 256
    lmax = int(3. * nside - 1.)
    bins = np.arange(10, lmax, 30)
    test = Experiment('test', lmax, freqs, beams=None)
    test.read_map_file('../data/test/maps/test_pl')
    bin_average = test.empirical_covmat
    ell_average = test.ell_mean
    factor = ell_average ** 2

    test_bin = Experiment('test', lmax, freqs, bins=bins, beams=None)
    test_bin.read_map_file('../data/test/maps/test_pl')
    bin_average_bin = test_bin.empirical_covmat
    ell_average_bin = test_bin.ell_mean
    factor_bin = ell_average_bin ** 2
    plt.figure()
    plt.plot(ell_average, factor * bin_average[0, 0, :], c='b', alpha=.5)
    plt.scatter(ell_average_bin, factor_bin * bin_average_bin[0, 0, :],
                c='k')
    plt.plot(ell_average, factor * bin_average[0, 1, :], c='orange',
             alpha=.5)
    plt.scatter(ell_average_bin, factor_bin * bin_average_bin[0, 1, :],
                c='k')
    plt.plot(ell_average, factor * bin_average[1, 0, :], c='g', alpha=.5)
    plt.scatter(ell_average_bin, factor_bin * bin_average_bin[1, 0, :],
                c='k')
    plt.plot(ell_average, factor * bin_average[1, 1, :], c='r', alpha=.5)
    plt.scatter(ell_average_bin, factor_bin * bin_average_bin[1, 1, :],
                c='k')
    plt.show()
