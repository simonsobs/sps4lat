"""
Optimization module
"""
import sys
import numpy as np
from functools import reduce
import scipy.optimize as opt


def optimizable_function(experiment, model, param_default):
    model.prepare_for_arrays(param_default)
    emp_cov = experiment.empirical_covmat

    def f(theta):
        model_cov = model.eval_array(theta)
        kl = kl_divergence(emp_cov, model_cov, experiment)
        return np.abs(kl)

    return f


def kl_divergence(emp_cov, model_cov, experiment):
    """
    Compute the KL divergence between the empirical covariance matrix and the
    modelled one
    Parameters
    ----------
    emp_cov : ndarray
        empirical covmat, shape is : ``(freqs,freqs,ells)``
    model_cov :ndarray
        modelled covmat, shape is : ``(freqs,freqs,ells)``
    experiment : Experiment
    Returns
    -------
    KL divergence : measure of the mismatch between the two matrices.

    """
    lmin_bins = np.arange(experiment.lmin, experiment.lmax, experiment.bin_size)
    lmax_bins = np.append(lmin_bins[1:], experiment.lmax)
    try:
        assert (emp_cov.shape[-2] == model_cov.shape[-2])
    except AssertionError:
        sys.exit('Empirical and modelled covmat have been computed at '
                 'different frequencies')

    try:
        assert (emp_cov.shape[-1] == model_cov.shape[-1] == len(lmin_bins))
    except AssertionError:

        sys.exit('Empirical and modelled covmat have been computed over a'
                 'different number of subdomains')
    n_bins = emp_cov.shape[-1]
    m = emp_cov.shape[1]
    k = np.zeros(n_bins)
    for i in range(n_bins):
        emp = emp_cov[..., i]
        mod = model_cov[..., i]
        Z = np.linalg.solve(np.linalg.cholesky(emp).T,
                            np.linalg.cholesky(mod).T).T
        k[i] = np.dot(Z.flatten().T, Z.flatten()) - 2. * np.sum(
            np.log(np.diag(Z))) - m
    # inv_emp = np.linalg.solve(emp,
    #                           np.broadcast_to(np.identity(m), emp.shape))
    # _, logdet = np.linalg.slogdet(
    #     np.einsum('lab,lbc->lac', inv_emp, mod))
    # kl = .5 * (np.einsum('lab,lba->l', inv_emp, mod) - logdet - m)
    weights = np.array(
        [(lmax_bins[i] - lmin_bins[i] + 1) * (lmax_bins[i] + lmin_bins[i - 1] + 1) for i in
         range(len(lmin_bins))])
    return (k.dot(weights)).sum()


def optimiser_test(experiment, model, param_start):
    f_optim = optimizable_function(experiment, model, param_start)
    theta_start = model.kwargs2array(param_start)
    res = opt.minimize(f_optim, x0=theta_start, tol=1e-5,
                       options={'maxiter': 10000})
    param_opt = model.array2kwargs(res.x)
    return param_opt
