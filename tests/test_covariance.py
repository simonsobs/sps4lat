import unittest
import numpy as np
import healpy as hp
from sps4lat import covariance as cov


class UtilsTest(unittest.TestCase):

    def setUp(self):
        self.nside = 256
        self.lmax = int(3. * self.nside - 1.)
        lmins = np.arange(0, self.lmax, 30)
        lmaxs = np.arange(29, self.lmax, 30)
        if lmaxs[-1] >= lmins[-1]:
            lmaxs = lmaxs[:-1]
        if lmaxs[-1] != self.lmax:
            lmaxs = np.append(lmaxs, self.lmax)
        self.domain_list = [cov.Domain(lmin, lmax) for (lmin, lmax) in
                            zip(lmins, lmaxs)]
        self.nbins = len(self.domain_list)
        a = np.random.rand(len(lmaxs), 10, 10)
        self.cov_kl = np.einsum('lab,lcb->lac', a, a)  # build pos.def matrix
        self.maps = [np.random.randn(hp.nside2npix(self.nside))
                     for _ in range(5)]
        self.alms = np.array([hp.map2alm(m) for m in self.maps])

    def test_get_covmat_alms(self):
        """ Test function that computes covmat from alms.
        Test for shape and type of ouput."""
        covmat = cov._get_covmat(self.alms, domain_list=self.domain_list)
        self.assertIsInstance(covmat, np.ndarray)
        self.assertEqual(covmat.shape, (self.nbins, 5, 5))

    def test_get_covmat_maps(self):
        """ Test function that computes covmat from maps.
        Test for shape and type of ouput."""
        covmat = cov._get_covmat_maps(self.maps,
                                      domain_list=self.domain_list)
        self.assertIsInstance(covmat, np.ndarray)
        self.assertEqual(covmat.shape, (self.nbins, 5, 5))

    def test_get_covmat_equals(self):
        """ Compare covmat computations from maps and alms.
        Difference should be small."""
        covmat_map = cov._get_covmat_maps(self.maps,
                                          domain_list=self.domain_list)
        covmat_alm = cov._get_covmat(self.alms,
                                     domain_list=self.domain_list)
        self.assertTrue(np.all(np.abs(covmat_map - covmat_alm) <= 1e-5))
        # BB : Seems arbitrary to choose 1e-5 ...


if __name__ == '__main__':
    unittest.main()
