import unittest
import numpy as np
import healpy as hp
from sps4lat import utils


class UtilsTest(unittest.TestCase):

    def setUp(self):
        self.nside = 1
        self.lmax = 3 * self.nside - 1
        self.cl = np.random.randn(self.lmax + 1)
        self.alms = hp.synalm(self.cl, new=True)
        self.maps = hp.alm2map(self.alms, nside=self.nside, verbose=False)
        self.alms_sorted = self.alms[[0, 1, 3, 2, 4, 5]]

    def test_sort_alms_ells(self):
        """ Test function that sorts the alms by ell. """
        alms_sorted = utils.sort_alms_ell(self.alms)
        self.assertIsInstance(alms_sorted, np.ndarray)
        self.assertTrue(np.all(alms_sorted[0] == self.alms_sorted))

    def test_get_alms(self):
        """ Test function that get alms from maps. """
        alms = utils.get_alms(self.maps, lmax=self.lmax)
        self.assertIsInstance(alms, np.ndarray)
        self.assertTrue(np.all(alms[0] - self.alms < 1e-2))
        ## (BB) : not satisfied with this test ... ##


if __name__ == '__main__':
    unittest.main()
