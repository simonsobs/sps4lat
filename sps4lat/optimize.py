"""
Optimization module
"""
import sys
import numpy as np
from functools import reduce
import scipy.optimize as opt


def optimizable_function(emp_cov, model, param_default):
    model.prepare_for_arrays(param_default)
    def f(theta):
        model_cov = model.eval_array(theta)
        kl = kl_divergence(emp_cov, model_cov)
        # print(theta)
        print(kl)
        return kl

    return f


def kl_divergence(emp_cov, model_cov):
    """
    Compute the KL divergence between the empirical covariance matrix and the
    modelled one
    Parameters
    ----------
    emp_cov : ndarray
        empirical covmat, shape is : ``(freqs,freqs,ells)``
    model_cov :ndarray
        modelled covmat, shape is : ``(freqs,freqs,ells)``
    Returns
    -------
    KL divergence : measure of the mismatch between the two matrices.

    """
    try:
        assert(emp_cov.shape == model_cov.shape)
    except AssertionError:
        print("Empirical covmat and modle ones don't have the same shape")
        return None
    n_bins = emp_cov.shape[-1]
    lmax = n_bins + 1
    k = np.zeros(n_bins)
    for i in range(n_bins):
        emp = emp_cov[..., i]
        mod = model_cov[..., i]
        m = emp.shape[1]
        Z = np.linalg.solve(np.linalg.cholesky(emp), np.linalg.cholesky(mod))
        k[i] = .5*(np.dot(Z.flatten(), Z.flatten().T) - 2. * np.sum(
            np.log(np.diag(Z))) - m)
    weights = 2.*np.linspace(2, lmax, lmax-1) + 1.
        # weights = np.array([(lmax_bins[i] - lmin_bins[i] + 1) * (
        #             lmax_bins[i] + lmin_bins[i - 1] + 1) for i in
        #                     range(len(lmin_bins))])
    return (k.dot(weights)).sum()


def optimiser_test(emp_cov, model, param_start):
    f_optim = optimizable_function(emp_cov, model, param_start)
    theta_start = model.kwargs2array(param_start)
    res = opt.minimize(f_optim, x0=theta_start,
                       options={'maxiter': 1000})
    param_opt = model.array2kwargs(res.x)
    return param_opt
