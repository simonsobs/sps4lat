"""
Optimization module
"""
import sys
import logging
import numpy as np
from functools import reduce
import scipy.optimize as opt
from scipy.signal import medfilt
import matplotlib.pyplot as plt


def optimizable_function(emp_cov, model, param_default, bins, delta_callback, fig):
    bins_max = bins[1:]
    bins_min = bins[:-1]
    n_bins = len(bins) - 1
    weights = np.array(
        [(bins_max[i] - bins_min[i]) * (bins_max[i] + bins_min[i]) for i in
         range(n_bins)])
    weights = bins_max * (2 * bins_max + 1) - (2 * bins_min + 1) * (2 * bins_min - 1) / 2

    cholesky_emp = np.linalg.cholesky(emp_cov.T)
    model.prepare_for_arrays(param_default)
    shape = cholesky_emp.shape
    id_3d = np.broadcast_to(np.identity(shape[1]), shape)

    n_calls = [0]
    x_old = [None]
    cov_old = [None]
    diff_old = [None]
    cholesky_mod_old = [None]
    
    ell = (bins[1:] + bins[:-1]) / 2
    to_dl = ell * (ell + 1) / 2 / np.pi
    f_samples = []
    g_samples = []
    

    def _update_old(x):
        """
        Check whether kl divergence and jac have already been computed for x.
        If not computes and caches them.
        :param x: ndarray
        :return:
        """
        # if x is different from the last one, we compute the model.
        if not np.all(x == x_old[0]):
            kwargs = model.array2kwargs(x)
            cov_old[0] = model.eval(**kwargs)
            diff_old[0] = model.diff_kwargs2array(model.diff(**kwargs))
            try:
                cholesky_mod_old[0] = np.linalg.cholesky(cov_old[0].T)
            except np.linalg.LinAlgError:
                print(kwargs)
            x_old[0] = x

    def _callback(x):
        if fig is None:
            logging.warning(f'Calls\t{n_calls[0]}\tKL:\t{f_optim(x):.2f}\tKL:\t{np.sum(diff_kl(x)**2):.2f}')
        else:
            f_samples.append(f_optim(x))
            g_samples.append(np.sum(diff_kl(x)**2))
            if len(f_samples) < 3:
                return
            fig.clear()
            n_freq = cov_old[0].shape[0]
            gs = fig.add_gridspec(n_freq, n_freq)
            assert n_freq != n_bins
            for i in range(n_freq):
                for j in range(i, n_freq):
                    ax = fig.add_subplot(gs[i, j])
                    ax.loglog(emp_cov[i, j] * to_dl)
                    ax.loglog(cov_old[0][i, j] * to_dl)
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
            
            ax = fig.add_subplot(gs[n_freq-2:n_freq, 0:2])
            ax.semilogy(medfilt(f_samples, 3))
            ax = fig.add_subplot(gs[n_freq-2:n_freq, 2:4])
            ax.semilogy(-np.diff(medfilt(f_samples, 3)))
            ax = fig.add_subplot(gs[n_freq-4:n_freq-2, 0:2])
            ax.semilogy(medfilt(g_samples, 3))
            #plt.tight_layout()
            fig.canvas.draw()
            print('DONE')

    def f_optim(x):
        _update_old(x)
        n_calls[0] = n_calls[0] + 1
        kl = kl_divergence(cholesky_emp, cholesky_mod_old[0], weights)
        return kl

    def diff_kl(x):
        if n_calls[0] % delta_callback == 0:
            _callback(x)

        _update_old(x)
        chol_inv = np.linalg.solve(cholesky_mod_old[0], id_3d)
        inv_model = np.einsum('lji,ljk->ikl', chol_inv, chol_inv)
        G = np.einsum('l,abl,bcl,cdl->adl', 0.5 * weights, inv_model,
                      cov_old[0] - emp_cov, inv_model)
        deriv = np.einsum('abl,pbal->p', G, diff_old[0])
        return deriv

    return f_optim, diff_kl


def kl_divergence(cholesky_emp, cholesky_mod, weights):
    """

    :param cholesky_emp: ndaray
         cholseky decomposition of the empircal cavmat (computed outside this
         function)
    :param cholesky_mod: ndarray
        chelseky decomposition of the model covmat (computed outside this
        function)
    :param weights
    :return: float : kl divergence
    """
    m = cholesky_emp.shape[1]
    Z = np.linalg.solve(cholesky_mod, cholesky_emp)
    # Z = np.linalg.solve(cholesky_emp, cholesky_mod)
    log_det = np.sum(np.log(np.diagonal(Z, axis1=1, axis2=2)), axis=-1)
    k = .5 * (np.einsum('lij,lij->l', Z, Z) - 2. * log_det - m)
    return (k.dot(weights)).sum()


def optimiser_test(emp_cov, model, param_start, bins, delta=20, deriv=True, fig=None, kwargs_opt={}):
    f_optim, jac = optimizable_function(emp_cov, model, param_start, bins, delta, fig)
    if not deriv:
        jac = None
    theta_start = model.kwargs2array(param_start)
    res = opt.minimize(f_optim, x0=theta_start, jac=jac, **kwargs_opt)
    param_opt = model.array2kwargs(res.x)
    return param_opt, res
