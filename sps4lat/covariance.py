""" module covariance
This module implements computation of the empirical covariance matrix.
"""
import sys
import numpy as np
from sps4lat import utils as utl
import os
import healpy as hp


class Experiment:
    def __init__(self, name, lmax, freqs, domain_list, beams=None):
        self.name = name
        self.lmax = lmax
        self.beams = beams
        self.freqs = freqs
        self.domain_list = domain_list

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
        return _get_covmat_maps(self.maps, self.domain_list, self.beams,
                                self.lmax)


def _get_covmat(alms_sorted, domain_list):
    """ Compute the empirical covmat on every subdomain in domain_list.
    :param domain_list: list of domains over which the empirical covariance
    will be computed
    :return: (Q*N*N) array (Q being the numbers of subdomains)

    Parameters
    ----------
    alms_sorted : ndarray
        array containing the alms at each frequency sorted by ell
    domain_list : list
        list of domains over which the empirical covariance will be computed
    Returns
    -------
    (Q*N*N) array (Q being the numbers of subdomains)
    """
    cov_mat_list = []
    for domain in domain_list:
        cov_mat_list.append(
            _get_covmat_subdomain(alms_sorted, domain.lmin, domain.lmax))
    return np.array(cov_mat_list)


def _get_covmat_maps(maps, domain_list, beams=None, lmax=None):
    """ Compute the empirical covmat on every subdomain in domain_list.

    Parameters
    ----------
    maps : list
        list of maps at different frequencies
    domain_list : list
        list of domains over which the empirical covariance will be computed
    beams : ndarray
        beams associated with each map (assumed gaussian)
    lmax : int
        max ell used to compute the covariance matrix
    Returns
    -------
    (Q*N*N) array (Q being the numbers of subdomains)
    """
    alms = utl.get_alms(maps, beams, lmax)
    alms_sorted = utl.sort_alms_ell(alms)
    return _get_covmat(alms_sorted, domain_list)


def _get_covmat_subdomain(alms_sorted, lmin, lmax):
    """ Compute the empirical covariance matrix between lmin and lmax.

    Parameters
    ----------
    alms_sorted : ndarray
        array containing the alms at each frequency sorted by ell
    lmin : int
        smallest ell of the subdomain (included)
    lmax : int
        largest ell of the subdomain (included)
    Returns
    -------
    covmat : ndarray
        empirical covariance over the sudomain. (N*N) array.

    """
    idx_start, idx_end = _get_subdomain(lmin, lmax)
    n_p = idx_end - idx_start
    covmat = 1. / n_p * np.einsum('is,js ->ij',
                                  alms_sorted[:, idx_start:idx_end],
                                  alms_sorted[:, idx_start:idx_end].conj())
    return covmat.real.astype(np.float64)  # TODO Converting to real (BB : is this the way to do it ?)


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


class Domain:
    def __init__(self, lmin, lmax):
        self.lmin = lmin
        self.lmax = lmax
        self.lmean = int((lmax + lmin) * .5)
