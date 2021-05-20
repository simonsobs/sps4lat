""" module model
This module defines a variety of models used as fg models.

BB : 23/04/20 for now the abstract class for a model is taken from
fgspectra. Might need to override that ?
"""
import numpy as np
from fgspectra.model import *
from fgspectra.cross import *
from fgspectra.power import *
from fgspectra.frequency import *


# class WhiteNoise(Model):
#     """White noise"""
#
#     def eval(self, nu=None, ell=None, nwhite=None):
#         """ Evaluation of the model
#
#         Parameters
#         ----------
#         nu : float or array
#             Frequency at which model will be evaluated. If array, the shape
#             is ``(freq)``.
#         ell : float or array
#             Multipoles at which model will be evaluated. If array, the shape
#             is ``(ells)``.
#         nwhite : ndarray
#             white noise levels, shape is ``(freqs)``
#         Returns
#         -------
#         cov : ndarray
#             Shape is ``(ells,freqs,freqs)``
#         """
#         if type(nu) in (float, int):
#             nu = [nu]
#         n_freqs = len(nu)
#         if type(ell) in (float, int):
#             ell = [ell]
#         n_ell = len(ell)
#         if type(nwhite) in (float, int):
#             nwhite = [nwhite]
#         if len(nwhite) == 1 and n_freqs > 1:
#             print('Expected {:d} noise levels but got 1. Will use the same '
#                   'at all frequencies'.format(n_freqs))
#             nwhite = np.ones(n_freqs) * nwhite
#         elif len(nwhite) != n_freqs:
#             print('Got {:d} white noise levels, expected {:d}'.format(
#                 len(nwhite), n_freqs))
#         res = np.broadcast_to(np.diag(nwhite**2), (n_ell, n_freqs, n_freqs))
#         return np.transpose(res, (1, 2, 0))
#
#
# class FreeSED(Model):
#     """Completely free SED."""
#
#     def eval(self, nu=None, sed=None):
#         """ Evaluation of the SED
#
#         Parameters
#         ----------
#         nu: float or array
#             Frequencies of the experiment
#         sed: float or array
#             SED at each frequency. Must be same size as nu.
#
#         Returns
#         -------
#         sed: ndarray
#             shape is ``(freqs)``
#         """
#         if type(sed) in (int, float):
#             sed = [sed]
#         if type(nu) in (int, float):
#             nu = [nu]
#         try:
#             assert len(nu) == len(sed)
#         except AssertionError:
#             print('Size of SED must match number of frequencies')
#             return None
#         return np.array(sed)
#
#
class CMB(Model):
    """Simple CMB model, using spectra in ../data/cmb.dat"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ell_read, cmb_tt_read = np.loadtxt("../data/cmb.dat", usecols=(0, 1),
                                           unpack=True)
        self.cmb_tt = np.concatenate((np.zeros(2), cmb_tt_read /
                                      (ell_read * (ell_read + 1) /
                                       2. / np.pi)))

    def eval(self, ell=None, ):
        """ Evaluation of the model

        Parameters
        ----------
        ell : float or array
            Multipoles at which model will be evaluated. If array, the shape
            is ``(ells)``.
        Returns
        -------
        cov : ndarray
            Shape is ``(ells)``
        """
        if type(ell) in (float, int):
            ell = [ell]

        res = self.cmb_tt[ell.astype('int')]
        return res
#
#
# class FreeCls(Model):
#     """
#     Model with free Cls.
#     """
#
#     def eval(self, ell=None, cls=None):
#         """ Evaluation of the model
#
#         Parameters
#         ----------
#         ell : float or array
#             Multipoles at which model will be evaluated. If array, the shape
#             is ``(ells)``.
#         cls : ndarray
#             cls of the component. Shape is ``(ells)``
#         Returns
#         -------
#         res : ndarray
#             Shape is ``(ells)``
#         """
#         if type(ell) in (float, int):
#             ell = [ell]
#         if type(cls) in (float, int):
#             cls = [cls]
#         try:
#             assert len(ell) == len(cls)
#         except AssertionError:
#             print('Cls must have same size as ells')
#         res = cls
#         return res
