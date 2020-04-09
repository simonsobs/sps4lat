""" module covariance
This module implements computation of the empirical covariance matrix.
"""
import numpy as np
from sps4lat import utils as utl

__all__ = ['get_covmat', 'get_covmat_maps']


def _get_covmat_subdomain(alms_sorted, lmin, lmax):
    """
    Compute the empirical covariance matrix on the subdomain defined by lmin
    and lmax.
    :param alms_sorted: array containing the alms at each frequency sorted by
    ells
    :param lmin: smallest ell of the subdomain (included)
    :param lmax: largest ell of the subdomain (included)
    :return: covmat : empirical covariance over that subdomain (N*N) array.
    """
    idx_start, idx_end = _get_subdomain(lmin, lmax)
    n_p = idx_end - idx_start
    covmat = 1. / n_p * np.einsum('is,js ->ij',
                                  alms_sorted[:, idx_start:idx_end],
                                  alms_sorted[:, idx_start:idx_end].conj())
    return covmat


def get_covmat(alms_sorted, domain_list):
    """
    Compute the empirical covmat on every subdomain in domain_list.
    :param alms_sorted: array containing the alms at each frequency sorted by
    ells
    :param domain_list: list of domains over which the empirical covariance
    will be computed
    :return: (Q*N*N) array (Q being the numbers of subdomains)
    """
    cov_mat_list = []
    for domain in domain_list:
        cov_mat_list.append(
            _get_covmat_subdomain(alms_sorted, domain.lmin, domain.lmax))
    return np.array(cov_mat_list)


def get_covmat_maps(map_list, domain_list, beams=None, lmax=None):
    """
    Compute the empirical covmat on every subdomain in domain_list.
    :param map_list: list of maps to compute empirical covmat from.
    :param domain_list: list of domains over which the empirical covariance
    will be computed
    :param beams : array of beams form each map
    :param lmax : ell up to which covariance matrix will be computed
    :return: (Q*N*N) array (Q being the numbers of subdomains)
    """
    alms = utl.get_alms(map_list, beams, lmax)
    alms_sorted = utl.sort_alms_ell(alms)
    return get_covmat(alms_sorted, domain_list)


def _get_subdomain(lmin, lmax):
    """
    Assuming alms sorted by ell, returns start and end indices such
    that alms[idx_start:idx_end] corrsponds to subdomain from lmin to lmax
    :param lmin: lmin of the subdomain (inclusive)
    :param lmax: lmax of the subdomain (inclusive)
    :return: idx_start, idx_end
    """
    idx_start = int(lmin * (lmin + 1) * .5)
    idx_end = int((lmax + 1) * (lmax + 2) * .5)
    return idx_start, idx_end


class Domain:
    def __init__(self, lmin, lmax):
        self.lmin = lmin
        self.lmax = lmax
        self.lmean = int((lmax + lmin) * .5)
