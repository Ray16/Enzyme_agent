"""Generic producer labels require exact independently fingerprinted atom inventories."""
import json
import numpy as np
import pytest
from structural_modeling_agent.validation import component_inventories,validate_pose,file_digest
from test_validation import modeling_record,execution_for,write_pose


def test_component_atom_apostrophes_survive_cif_decoding_and_source_changes_fail(tmp_path):
    source=tmp_path/'component.cif';source.write_text('data_SYN\n_chem_comp.id SYN\nloop_\n_chem_comp_atom.atom_id\n_chem_comp_atom.type_symbol\n"C1\'" C\nP1 P\nH1 H\n')
    refs={'SYN':{'path':str(source),'sha256':file_digest(source)}}
    assert component_inventories(refs)=={'SYN':{"C1'":'C','P1':'P'}}
    source.write_text(source.read_text()+'O1 O\n')
    with pytest.raises(ValueError,match='fingerprint'):component_inventories(refs)


def test_generic_cofactor_name_is_never_accepted_without_matching_atoms_and_elements(tmp_path):
    import gemmi
    record=modeling_record(tmp_path);record['cofactors']=[{'ccd':'SYN','role':'cosubstrate','count':1}]
    execution=execution_for(record)
    # Use a separate cofactor chain; the geometry is irrelevant to this identity assertion.
    execution['entity_map'].append({'chain':'X','kind':'cofactor','ccd':'SYN','role':'cosubstrate'})
    pose=write_pose(tmp_path/'pose.pdb');st=gemmi.read_structure(str(pose));chain=gemmi.Chain('X');res=gemmi.Residue();res.name='LIG';res.seqid=gemmi.SeqId(1,' ');res.het_flag='H'
    atom=gemmi.Atom();atom.name='P1';atom.element=gemmi.Element('P');atom.pos=gemmi.Position(10,0,0);res.add_atom(atom);chain.add_residue(res);st[0].add_chain(chain);st.write_pdb(str(pose))
    def verdict(inventory):
        result=validate_pose(pose,execution,component_atoms={'SYN':inventory},generic_ligand_names=['LIG'])
        return next(c['status'] for c in result['checks'] if c['check']=='identity.X')
    assert verdict({'P1':'P'})=='pass'
    assert verdict({'P1':'MG'})=='fail'
    assert verdict({'P1':'P','O1':'O'})=='fail'
