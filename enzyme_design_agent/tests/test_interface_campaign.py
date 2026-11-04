"""Same-pose contact and fresh-sequence alignment regression tests."""
import tempfile
from pathlib import Path
import unittest
from enzyme_design_agent.saturation.interface import interface_positions
from enzyme_design_agent.saturation.msa import alignment_csv


def atom(record, serial, name, resname, chain, rn, x, element):
    return f'{record:<6}{serial:5d} {name:^4s} {resname:>3s} {chain}{rn:4d}    {x:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{90.:6.2f}          {element:>2s}\n'


class InterfaceTest(unittest.TestCase):
    def write(self, p, x, a, b):
        p.write_text(atom('ATOM',1,'CA','ALA','A',1,x,'C')+
                     atom('HETATM',2,'P','LIG','B',1,a,'P')+
                     atom('HETATM',3,'O1','LIG','C',1,b,'O'))
    def test_contacts_must_cooccur(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.pdb',Path(d)/'b.pdb'
            self.write(a,0,1,20);self.write(b,0,20,1)
            args=([a,b],'A',{'chain':'B'},{'chain':'C','atom_names':['O1']})
            self.assertEqual(interface_positions(*args),[])
            self.write(b,0,5,5)
            rows=interface_positions(*args)
            self.assertEqual(rows[0]['n_contact_poses'],1)
            self.assertEqual(rows[0]['contact_fraction'],.5)
            self.assertEqual(interface_positions(*args,min_contact_poses=2),[])
    def test_missing_atom_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.pdb';self.write(p,0,1,1)
            with self.assertRaises(ValueError):
                interface_positions([p],'A',{'chain':'B'}, {'chain':'C','atom_names':['O2']})
    def test_msa_query_identity_and_multiline(self):
        self.assertIn('-1,ACD\n',alignment_csv('>q\nAC\nD\n>h\nA-D\n','ACD'))
        with self.assertRaises(ValueError):alignment_csv('>q\nACD\n','ACE')
        with self.assertRaises(ValueError):alignment_csv('>q\nACD\n>h\nAC\n','ACD')

if __name__=='__main__':unittest.main()

class CampaignTest(unittest.TestCase):
    def test_prepare_interface_and_reject_changed_inputs(self):
        import json
        from enzyme_design_agent.saturation.campaign import prepare
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'source.pdb'
            InterfaceTest().write(p,0,1,2)
            spec={'schema_version':'saturation-1','selection_evidence':'Explicit user shortlist',
                  'metrics':{'alignment':'user supplied'},'n_samples':50,
                  'candidates':[{'id':'candidate_x','selection_reason':'Retrieved evidence',
                      'assembly_status':'specified monomer','structure_path':str(p),
                      'sequence':'A','protein_chain':'A','anchor':'B','components':['B','C'],
                      'ensemble_paths':[str(p)],'interface':{'fragment_a':{'chain':'B'},'fragment_b':{'chain':'C'}},
                      'boltz_input':{'version':1,'sequences':[{'protein':{'id':'A','sequence':'A'}}]}}]}
            sp=root/'spec.json';sp.write_text(json.dumps(spec))
            m=prepare(sp,root/'out')
            self.assertEqual(m['n_variants'],20)
            self.assertEqual(m['expected_models'],1000)
            self.assertEqual(prepare(sp,root/'out')['n_variants'],20)
            p.write_text(p.read_text()+'END\n')
            with self.assertRaises(ValueError):prepare(sp,root/'out')

class MetricTest(unittest.TestCase):
    def test_angle_invariant_to_distance_scaling(self):
        from structural_modeling_agent.proxies.ensemble_geometry import proximity_weighted_angle_and_distance as score
        a=score([[3.,150.],[6.,90.]])
        b=score([[6.,150.],[12.,90.]])
        self.assertAlmostEqual(a[0],130.)
        self.assertAlmostEqual(a[1],4.)
        self.assertAlmostEqual(a[0],b[0])
        self.assertAlmostEqual(a[1]*2,b[1])
        with self.assertRaises(ValueError):score([[0.,150.]])
