import unittest
import numpy as np
import torch
from forte.spatial import pool_regions,aggregate_regions

class SpatialTests(unittest.TestCase):
    def test_quadrants_preserve_location(self):
        a=torch.arange(16,dtype=torch.float32).reshape(16,1)
        regions=pool_regions(a,torch.tensor([1,8,8]),merge=2)
        np.testing.assert_allclose(regions[:,0].numpy(),[2.5,4.5,10.5,12.5,7.5])

    def test_motion_ignores_constant_scene_offset(self):
        x=np.arange(30,dtype=np.float32).reshape(3,5,2)
        np.testing.assert_allclose(aggregate_regions(x,'motion'),aggregate_regions(x+100,'motion'))

    def test_bad_grid_is_rejected(self):
        with self.assertRaises(ValueError):pool_regions(torch.zeros(7,2),[1,8,8])
