"""
Optimization module
"""
import sys
import numpy as np
from functools import reduce
import scipy.optimize as opt


def _theta2param(param_default, theta):
    """
    Function that takes a parameter vector theta and returns a dict,
    readable by a model.
    Parameters
    ----------
    param_default : dict
        Default parameters of the models (NOT fixed ones)
    theta : ndarray
        Parameter vector, used in optimization routines
    Returns
    -------
    param : dict
        Parameter dict, user friendly and readable by model.eval()
    """
    param_names = [kk for kk in param_default.keys()]
    param_types = [type(vv) for vv in param_default.values()]
    param_shapes = [vv.shape if type(vv) != float else 1 for vv in
                    param_default.values()]
    param_values = []
    count = 0
    n_params = len(param_names)
    for ii in range(n_params):
        typ = param_types[ii]
        if typ == float:
            value = theta[count]
            count += 1
        else:
            shape = param_shapes[ii]
            n_elem = reduce((lambda x, y: x * y), shape)
            value = theta[count:count + n_elem].reshape(shape)
            count += n_elem
        param_values.append(value)
    params = dict(zip(param_names, param_values))
    return params


def _param2theta(param_default, param):
    """
    Function that takes a parameter dict and returns a parameter vector,
    passed to optimization routines.
    Parameters
    ----------
    param_default : dict
        Default parameters of the models (NOT fixed ones)
    param : dict
        Parameter dict, user friendly and readable by model.eval()
    Returns
    -------
    theta : ndarray
        Parameter vector, used in optimization routines
    """
    try:
        assert [kk for kk in param_default.keys()] == [kk for kk in
                                                       param.keys()]
    except AssertionError:
        print('Order not the same ... ')
    theta_list = [np.array(vv).flatten() for vv in param.values()]
    return np.concatenate(theta_list)


def optimizable_function(experiment, model, param_default):
    emp_cov = experiment.empirical_covmat
    freqs = experiment.freqs
    ells = np.array([domain.lmean for domain in experiment.domain_list])
    def f(theta):
        param = _theta2param(param_default, theta)
        model_cov = model.eval(nu=freqs, ell=ells, **param)
        kl = kl_divergence(model_cov, emp_cov, experiment.domain_list)
        return np.abs(kl)

    return f


def kl_divergence(emp_cov, model_cov, domain_list):
    """
    Compute the KL divergence between the empirical covariance matrix and the
    modelled one
    Parameters
    ----------
    emp_cov : ndarray
        empirical covmat, shape is : ``(ells,freqs,freqs)``
    model_cov :ndarray
        modelled covmat, shape is : ``(ells,freqs,freqs)``
    domain_list : list
        List of subdomains across which both matrices have been computed
    Returns
    -------
    KL divergence : measure of the mismatch between the two matrices.

    """
    try:
        assert (emp_cov.shape[1] == model_cov.shape[1])
    except AssertionError:
        sys.exit('Empirical and modelled covmat have been computed at '
                 'different frequencies')

    try:
        assert (emp_cov.shape[0] == model_cov.shape[0] == len(domain_list))
    except AssertionError:
        sys.exit('Empirical and modelled covmat have been computed over a'
                 'different number of subdomains')
    m = emp_cov.shape[1]
    inv_emp_cov = np.linalg.solve(emp_cov,np.broadcast_to(np.identity(m),emp_cov.shape))
    _, logdet = np.linalg.slogdet(
        np.einsum('lab,lbc->lac', inv_emp_cov, model_cov))
    kl = .5 * (np.einsum('lab,lba->l', inv_emp_cov, model_cov) - logdet - m)
    weights = np.array([d.lmax - d.lmin + 1 for d in domain_list])
    # BB : think this is wrong, should take number of modes, not number of
    # weights = np.array(
    #     [(d.lmax - d.lmin + 1) * (d.lmax + d.lmin + 1) for d in domain_list])
    return (weights * kl).sum()


def optimiser_test(experiment, model, param_start):
    param = model.free_parameters
    print('Optimization run on {:s}'.format(' '.join(param.keys())))
    f_optim = optimizable_function(experiment, model, param_start)
    theta_start = _param2theta(param_start, param_start)
    res = opt.minimize(f_optim, x0=theta_start, tol=1e-6,options={'maxiter':1000})
    param_opt = _theta2param(param_start, res.x)
    return param_opt
