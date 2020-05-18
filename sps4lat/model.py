""" module model
This module defines a variety of models used as fg models.

BB : 23/04/20 for now the abstract class for a model is taken from
fgspectra. Might need to override that ?
"""
import numpy as np
from fgspectra import model as fgm
from fgspectra import cross as fgc
from fgspectra import power as fgp
from fgspectra import frequency as fgf


class PowerLaw(fgm.Model):
    """Simple power law in frequency and power"""

    def __init__(self, **kwargs):
        self.set_defaults(**kwargs)

    def eval(self, nu=None, ell=None, nu_0=None, ell_0=None, beta=None,
             alpha=None):
        """ Evaluation of the model

        Parameters
        ----------
        nu : float or array
            Frequency at which model will be evaluated. If array, the shape
            is ``(freq)``.
        ell : float or array
            Multipoles at which model will be evaluated. If array, the shape
            is ``(ells)``.
        nu_0 : float
            Reference frequency
        ell_0 : float
            Reference scale
        beta : float
            Spectral index of the SED
        alpha : float
            Spectral index of the Cl
        Returns
        -------
        cov : ndarray
            Shape is ``(ells,freqs,freqs)``
        """
        fg = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
        freqs_param = dict(nu=nu, nu_0=nu_0, beta=beta)
        ells_param = dict(ell=ell, ell_0=ell_0, alpha=alpha)
        cov = np.transpose(fg(freqs_param, ells_param), (2, 0, 1))
        return cov

    @property
    def free_parameters(self):
        """
        Returns
        -------
        dict containing free parameters of the model
        """
        return {k: v for k, v in self.defaults.items()
                if v is None and k not in ['ell', 'nu']}

    @property
    def fixed_parameters(self):
        """
        Returns
        -------
        dict containing fixed parameters of the model ie, one that are
        specified when model is called
        """
        return {k: v for k, v in self.defaults.items() if v is not None}


class Sum(fgc.Sum):
    def eval(self, *seq, **kwseq):
        """ Returns the sum of the covariance matrices.

        Parameters
        ----------
        kwseq : sequence or dict
            Either a sequence of dicts where `kwseq[i]`` is a dictionary
            containing the keyword arguments of the ``i``-th cross-spectrum.
            Or a single dict in which case parameters are passed to the
            relevant models.

        Returns
        -------
        cov : ndarray
            Shape is ``(ells,freqs,freqs)``
        """
        if seq:
            crosses = (cross(**kwargs) for cross, kwargs in
                       zip(self._crosses, seq))
        else:
            params = []
            for cross in self._crosses:
                params.append(
                    {k: kwseq[k] for k, v in cross.defaults.items() if
                     v is None})
            crosses = (cross(**kwargs) for cross, kwargs in
                       zip(self._crosses, params))
        res = next(crosses)
        for cross_res in crosses:
            res = res + cross_res
        return res

    @property
    def free_parameters(self):
        """
        Returns
        -------
        dict containing free parameters of the model
        """
        free = {}
        for cross in self._crosses:
            free = {**free, **cross.free_parameters}
        return free

    @property
    def fixed_parameters(self):
        """
        Returns
        -------
        dict containing fixed parameters of the model ie, one that are
        specified when model is called
        """
        fixed = {}
        for cross in self._crosses:
            fixed = {**fixed, **cross.fixed_parameters}
        return fixed


class WhiteNoise(fgm.Model):
    """White noise"""

    def __init__(self, **kwargs):
        self.set_defaults(**kwargs)

    def eval(self, nu=None, ell=None, nwhite=None):
        """ Evaluation of the model

        Parameters
        ----------
        nu : float or array
            Frequency at which model will be evaluated. If array, the shape
            is ``(freq)``.
        ell : float or array
            Multipoles at which model will be evaluated. If array, the shape
            is ``(ells)``.
        nwhite : ndarray
            white noise levels, shape is ``(freqs)``
        Returns
        -------
        cov : ndarray
            Shape is ``(ells,freqs,freqs)``
        """
        if type(nu) in (float, int):
            nu = [nu]
        n_freqs = len(nu)
        if type(ell) in (float, int):
            ell = [ell]
        n_ell = len(ell)
        if type(nwhite) in (float, int):
            nwhite = [nwhite]
        if len(nwhite) == 1 and n_freqs > 1:
            print('Expected {:d} noise levels but got 1. Will use the same '
                  'at all frequencies'.format(n_freqs))
            nwhite = np.ones(n_freqs) * nwhite
        elif len(nwhite) != n_freqs:
            print('Got {:d} white noise levels, expected {:d}'.format(
                len(nwhite), n_freqs))
        cov = np.zeros((n_ell, n_freqs, n_freqs)) + np.broadcast_to(
            np.diag(nwhite), (n_ell, n_freqs, n_freqs))

        return cov

    @property
    def free_parameters(self):
        """
        Returns
        -------
        dict containing free parameters of the model
        """
        return {k: v for k, v in self.defaults.items()
                if v is None and k not in ['ell', 'nu']}

    @property
    def fixed_parameters(self):
        """
        Returns
        -------
        dict containing fixed parameters of the model ie, one that are
        specified when model is called
        """
        return {k: v for k, v in self.defaults.items() if v is not None}


class CMB(fgm.Model):
    """Simple CMB model, using spectra in ../data/cmb.dat"""

    def __init__(self, **kwargs):
        self.set_defaults(**kwargs)

    def eval(self, nu=None, ell=None, a_sed=None):
        """ Evaluation of the model

        Parameters
        ----------
        nu : float or array
            Frequency at which model will be evaluated. If array, the shape
            is ``(freq)``.
        ell : float or array
            Multipoles at which model will be evaluated. If array, the shape
            is ``(ells)``.
        a_sed : ndarray
            Frequency calibration of the CMB, shape is ``(freqs)``
        Returns
        -------
        cov : ndarray
            Shape is ``(ells,freqs,freqs)``
        """
        if type(nu) in (float, int):
            nu = [nu]
        if type(ell) in (float, int):
            ell = [ell]
        if type(a_sed) in (float, int):
            a_sed = [a_sed]
        try:
            assert len(nu) == len(a_sed)
        except AssertionError:
            print(
                "Got {:d} calibrations, expected {:d} ! ".format(len(a_sed),
                                                                 len(nu)))
        ell_read, cmb_tt_read = np.loadtxt("../data/cmb.dat", usecols=(0, 1),
                                           unpack=True)
        cmb_tt = np.concatenate((np.zeros(2), cmb_tt_read
                                 / (ell_read * (ell_read + 1) / 2. / np.pi)))
        cov = np.einsum('i,l,j->lij', a_sed, cmb_tt[ell.astype('int')],
                        a_sed)
        return cov

    @property
    def free_parameters(self):
        """
        Returns
        -------
        dict containing free parameters of the model
        """
        return {k: v for k, v in self.defaults.items()
                if v is None and k not in ['ell', 'nu']}

    @property
    def fixed_parameters(self):
        """
        Returns
        -------
        dict containing fixed parameters of the model ie, one that are
        specified when model is called
        """
        return {k: v for k, v in self.defaults.items() if v is not None}


if __name__ == '__main__':
    cmb = CMB()
    ell_cmb = np.linspace(2, 1000, 999)
    nl = WhiteNoise()
    dict_cmb = dict(nu=np.array([100., 200.]), ell=ell_cmb,
                    a_sed=np.array([1., 2.]))
    dict_nl = dict(nu=np.array([100., 200.]), ell=ell_cmb,
                   nwhite=np.array([1., 2.]))
    sm = Sum(cmb, nl)
    cov_cmb = sm.eval(dict_cmb, dict_nl)
    print(cov_cmb.shape)
