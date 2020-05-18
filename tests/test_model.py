import unittest
import numpy as np
from sps4lat import model as mod


class UtilsTest(unittest.TestCase):

    def setUp(self):
        self.ell_0 = 1000
        self.nu_0 = 200.
        self.alpha = 2.
        self.beta = 3.
        self.ell = np.linspace(0, 1000, 1001)
        self.nu = np.array([100., 200., 300., 400.])
        self.nwhite = np.arange(4)
        self.a_sed = np.ones(4)

    def test_models(self):
        pl = mod.PowerLaw(ell_0=self.ell_0, nu_0=self.nu_0)
        nl = mod.WhiteNoise()
        cmb = mod.CMB()
        cov_pl = pl.eval(nu=self.nu, ell=self.ell, alpha=self.alpha,
                         beta=self.beta)
        cov_nl = nl.eval(nu=self.nu, ell=self.ell, nwhite=self.nwhite)
        cov_cmb = cmb.eval(nu=self.nu, ell=self.ell, a_sed=self.a_sed)
        self.assertIsInstance(cov_pl, np.ndarray)
        self.assertEqual(cov_pl.shape, (1001, 4, 4))
        self.assertIsInstance(cov_nl, np.ndarray)
        self.assertEqual(cov_nl.shape, (1001, 4, 4))
        self.assertIsInstance(cov_cmb, np.ndarray)
        self.assertEqual(cov_cmb.shape, (1001, 4, 4))

    def test_sum(self):
        pl = mod.PowerLaw(ell_0=self.ell_0, nu_0=self.nu_0)
        nl = mod.WhiteNoise()
        cmb = mod.CMB()
        sm = mod.Sum(pl, nl, cmb)
        cov_sum = sm.eval(nu=self.nu, ell=self.ell, alpha=self.alpha,
                          beta=self.beta, nwhite=self.nwhite,
                          a_sed=self.a_sed)
        self.assertIsInstance(cov_sum, np.ndarray)
        self.assertEqual(cov_sum.shape, (1001, 4, 4))
        dict_pl = dict(nu=self.nu, ell=self.ell, alpha=self.alpha,
                       beta=self.beta)
        dict_nl = dict(nu=self.nu, ell=self.ell, nwhite=self.nwhite)
        dict_cmb = dict(nu=self.nu, ell=self.ell, a_sed=self.a_sed)
        cov_sum_seq = sm.eval(dict_pl, dict_nl, dict_cmb)
        self.assertIsInstance(cov_sum_seq, np.ndarray)
        self.assertEqual(cov_sum_seq.shape, (1001, 4, 4))
        self.assertTrue(np.all(np.abs(cov_sum_seq - cov_sum) <= 1e-10))


if __name__ == '__main__':
    unittest.main()
