import unittest
import numpy as np
import healpy as hp
from sps4lat import preprocessing as pre


class UtilsTest(unittest.TestCase):

    def setUp(self):
        self.nside = 128
        self.lmax = int(3. * self.nside - 1.)
        N_freqs = 3
        self.beams = np.ones(N_freqs)
        self.maps = [np.random.randn(hp.nside2npix(self.nside))
                     for _ in range(N_freqs)]
        self.alms = np.array([hp.map2alm(m) for m in self.maps])

    def test_alms_from_maps(self):
        """Test of function that computes alms from healpix maps."""
        alms = pre.alms_from_maps(self.maps)
        alms_beams = pre.alms_from_maps(self.maps, self.beams)
        self.assertIsInstance(alms, np.ndarray)
        self.assertIsInstance(alms_beams, np.ndarray)

    def test_empirical_covmat(self):
        """Test of function that computes empirical covmat."""
        covmat = pre.empirical_covmat(self.alms)
        covmat_lmax = pre.empirical_covmat(self.alms, lmax=200)
        self.assertIsInstance(covmat, np.ndarray)
        self.assertIsInstance(covmat_lmax, np.ndarray)



if __name__ == '__main__':
    unittest.main()
