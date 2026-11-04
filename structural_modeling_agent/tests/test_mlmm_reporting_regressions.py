"""Last-stage ML/MM report robustness; no engine calculations are invoked."""
import dataclasses
import json

import pytest

from structural_modeling_agent.mlmm import NotReportable
from test_mlmm_verdict import clean_profile


def test_rejected_profile_does_not_serialize_derived_barriers():
    profile=clean_profile(reaction_mode_overlap=.04)
    with pytest.raises(NotReportable): profile.barrier()
    result=profile.to_dict()
    assert not result["reportable"]
    assert result["barrier_kcal"] is None and result["reaction_energy_kcal"] is None
    assert [s["energy_kcal"] for s in result["states"]]==[0,15,-32]


@pytest.mark.parametrize("field,value",[("reaction_mode_overlap",float("nan")),("reaction_mode_overlap",float("inf")),
    ("reaction_mode_overlap",1.01),("max_force_ev_per_a",float("nan")),("max_force_ev_per_a",-.1),
    ("energy_kcal",float("nan")),("energy_kcal",float("inf")),("n_imag",True),("n_atoms",0),
    ("imaginary_cm",(float("nan"),)),("driven_coordinates",{"C-O":float("nan")})])
def test_nonfinite_or_malformed_fields_cannot_authorize_reporting(field,value):
    profile=clean_profile()
    if field=="reaction_mode_overlap": setattr(profile,field,value)
    else: profile.transition_state=dataclasses.replace(profile.transition_state,**{field:value})
    assert not profile.verdict().ok
    with pytest.raises(NotReportable): profile.barrier()
    result=profile.to_dict()
    assert not result["reportable"] and result["barrier_kcal"] is None
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize("state_name",["reactant","product"])
def test_all_minima_need_real_mode_evidence(state_name):
    profile=clean_profile()
    setattr(profile,state_name,dataclasses.replace(getattr(profile,state_name),n_vib=None))
    report=profile.verdict()
    assert not report.ok
    assert next(c for c in report.checks if c.name=="mode_count_consistency").status=="ABSTAIN"
    assert profile.to_dict()["reaction_energy_kcal"] is None


def test_hessian_scope_can_be_declared_independently_of_energy_atom_scope():
    profile=clean_profile()
    profile.transition_state=dataclasses.replace(profile.transition_state,mode_atom_count=10)
    assert not profile.verdict().ok  # 161 modes cannot fit a declared ten-atom Hessian.
