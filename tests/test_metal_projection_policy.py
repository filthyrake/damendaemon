import unittest
from damendaemon.metal_projection_policy import use_packed_projection


class MetalProjectionPolicyTests(unittest.TestCase):
    def test_row_precision_and_matrix_size_boundaries(self):
        for fp32 in (True,False):
            self.assertTrue(use_packed_projection(4,896,896,fp32))
        for rows in (5,8):
            self.assertTrue(use_packed_projection(rows,896,896,True))
            self.assertFalse(use_packed_projection(rows,896,896,False))
        for rows in (9,16):
            self.assertFalse(use_packed_projection(rows,896,896,True))
            self.assertTrue(use_packed_projection(rows,896,4864,True))
            self.assertFalse(use_packed_projection(rows,896,4864,False))
        self.assertFalse(use_packed_projection(17,896,4864,True))

    def test_large_offsets_keep_expansion_path(self):
        self.assertFalse(use_packed_projection(8,2**27,32,True))
        self.assertFalse(use_packed_projection(8,32,2**27,True))
