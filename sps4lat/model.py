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


class CMB(Model):
    """Simple CMB model, using spectra in ../data/cmb.dat"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ell_read, cmb_tt_read = np.loadtxt("../data/cmb.dat", usecols=(0, 1),
                                           unpack=True)
        # self.cmb_tt = np.concatenate((np.zeros(2), cmb_tt_read /
        #                               (ell_read * (ell_read + 1) /
        #                                2. / np.pi)))
        self.cmb_tt = np.concatenate((np.zeros(2), cmb_tt_read ))

    def eval(self, ell=None, amp=1.):
        """ Evaluation of the model

        Parameters
        ----------
        ell : float or array
            Multipoles at which model will be evaluated. If array, the shape
            is ``(ells)``.
        amp : float
            Amplitude used to rescale the spectra
        Returns
        -------
        cov : ndarray
            Shape is ``(ells)``
        """
        if type(ell) in (float, int):
            ell = [ell]

        res = self.cmb_tt[ell.astype('int')]
        return amp*res

    def diff(self, **kwargs):
        if 'ell' in kwargs:
            raise NotImplementedError(
                'Derivatives with respect to ell not implemented')

        defaults = self.defaults

        if defaults['amp'] is not None:
            return {}

        amp = np.asarray(kwargs['amp'])
        ell = defaults['ell']
        res = np.zeros((amp.size, amp.size, ell.size))

        np.einsum('aal->al', res)[:] = self.eval(ell=ell, amp=1.)
        res_amp = res.reshape((amp.size,) + amp.shape + ell.shape)

        return {'amp': res_amp}



class GroundBasedNoise(Model):
    """White + atmospheric noise"""

    def eval(self, nu=None, ell=None, nwhite=None, nred=None, ell_knee=None,
             alpha_knee=None):
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
        nred : ndarray
            red noise level, shape is ``(freqs)``
        ell_knee : float
            pivot scale assumed the same for all freqs
        alpha_knee : float
            scale factor, assumed the same for all freqs
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
        if type(nred) in (float, int):
            nred = [nred]
        N = len(nu)
        noise = np.zeros((N, N, n_ell))
        for i in range(N):
            noise[i, i, :] = nred[i] * (ell / ell_knee) ** alpha_knee + \
                             nwhite[i]**2
        return np.einsum('ijl,l->ijl', noise, ell*(ell+1)/2./np.pi)

    def diff(self, nu=None, ell=None, nwhite=None, nred=None, ell_knee=None,
             alpha_knee=None):
        """ Evaluation of the derivative of the model

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
        diff: dict
            Each key corresponds to the the derivative with respect to a parameter.
        """
        (nu, ell, nwhite, red, ell_knee,
         alpha_knee) = self._replace_none_args(
            (nu, ell, nwhite, nred, ell_knee, alpha_knee))
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
        if type(nred) in (float, int):
            nred = [nred]
        diff_nwhite_ell = np.zeros((n_freqs, n_freqs, n_freqs))
        np.fill_diagonal(diff_nwhite_ell, 2. * nwhite)
        diff_nwhite = np.broadcast_to(diff_nwhite_ell,
                                      (n_ell, n_freqs, n_freqs, n_freqs)).T
        return {'nwhite': diff_nwhite}
