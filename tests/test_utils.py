import unittest
import numpy as np
import healpy as hp
from sps4lat import utils


class UtilsTest(unittest.TestCase):

    def setUp(self):
        self.alm_2_sort = np.random.randn(5, hp.Alm.getsize(lmax=5))
        idx_sorted = [0, 1, 6, 2, 7, 11, 3, 8, 12, 15, 4, 9, 13, 16, 18, 5,
                      10, 14, 17, 19, 20]
        self.alm_sorted = self.alm_2_sort[:, idx_sorted]
        nside_list = [16, 32, 64, 128]
        self.maps = [np.random.randn(hp.nside2npix(nside=nside)) for nside in
                     nside_list]
        self.lmax = 3. * 16. - 1.

    def test_sort_alms_ells(self):
        """ Test function that sorts the alms by ell. """
        alms_sorted = utils.sort_alms_ell(self.alm_2_sort)
        self.assertIsInstance(alms_sorted, np.ndarray)
        self.assertTrue(np.all(alms_sorted == self.alm_sorted))

    def test_get_alms(self):
        """ Test function that get alms from maps. """
        alms = utils.get_alms(self.maps)
        self.assertIsInstance(alms, np.ndarray)
        self.assertSetEqual(set([hp.Alm.getlmax(len(alm)) for alm in alms]),
                            {self.lmax})
        with self.assertRaises(SystemExit):
            _ = utils.get_alms(self.maps, lmax=100)

if __name__ == '__main__':
    unittest.main()
