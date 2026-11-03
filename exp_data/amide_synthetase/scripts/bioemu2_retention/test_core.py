"""Check rigid-motion invariance and sensitivity to ligand departure."""
import unittest
import numpy as np
from core import measure


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.xyz=np.array([[0,0,0],[3,0,0],[0,3,0],[0,0,3],
                           [1,1,1],[1,1,2],[2,1,1],[1,2,1]],dtype=float)
        self.kw=dict(protein=[0,1,2,3],ligand=[4,5],residue_ids=[0,1,2,3,4,4,5,6],
            ca=[0,1,2,3],atp_pa=6,atp_bridge=7,ligand_oxygens=[4,5],mg=[7])

    def test_rigid_motion_does_not_look_like_departure(self):
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        r=measure(self.xyz,self.xyz@rotation+np.array([40,20,10]),**self.kw)
        self.assertAlmostEqual(r['pocket_centroid_displacement_A'],0,places=10)
        self.assertEqual(r['native_contact_retention'],1)
        self.assertTrue(r['pocket_retained'])

    def test_departure_loses_contacts_and_moves_centroid(self):
        moved=self.xyz.copy()
        moved[[4,5]]+=20
        r=measure(self.xyz,moved,**self.kw)
        self.assertEqual(r['native_contact_retention'],0)
        self.assertGreater(r['pocket_centroid_displacement_A'],30)
        self.assertFalse(r['pocket_retained'])


if __name__=='__main__':
    unittest.main()
