import unittest
import numpy as np
import torch
from forte.temporal_model import transform, TemporalDetector, clip_logit

class IdentityPCA:
    def transform(self,x):return x

class TemporalTests(unittest.TestCase):
    def test_reversing_time_reverses_change(self):
        x=np.array([[1,0],[1,1],[0,1],[-1,1]],dtype=np.float32)
        forward=transform(x,IdentityPCA())
        backward=transform(x[::-1],IdentityPCA())[::-1]
        np.testing.assert_allclose(forward[:,:4],backward[:,:4],atol=1e-6)
        np.testing.assert_allclose(forward[:,4:],-backward[:,4:],atol=1e-6)

    def test_single_frame_is_finite(self):
        x=transform(np.zeros((1,32),dtype=np.float32),IdentityPCA())
        out=TemporalDetector().eval()(torch.from_numpy(x))
        self.assertEqual(tuple(out.shape),(1,))
        self.assertTrue(torch.isfinite(clip_logit(out)))
