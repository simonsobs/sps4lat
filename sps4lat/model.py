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
        """
        Evaluation of the model
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


if __name__ == '__main__':
    pl = PowerLaw(nu_0=150., ell_0=1500)
    print(pl.free_parameters)
    print(pl.fixed_parameters)
    cov_model = pl(np.array([100., 200.]), np.linspace(2, 100, 99),
                   beta=2.5, alpha=3.)
