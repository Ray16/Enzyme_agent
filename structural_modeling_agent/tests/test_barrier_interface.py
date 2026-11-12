import json

from test_modeling_agent import make_agent
from structural_modeling_agent.agent import tool_schemas


def test_structural_agent_owns_barrier_tools(tmp_path):
    agent=make_agent(tmp_path)
    names={s['function']['name'] for s in tool_schemas()}
    assert {'inspect_barrier_contract','prepare_barrier_inputs','inspect_barrier_result'}<=names
    assert agent.dispatch('inspect_barrier_contract',{})['owner']=='structural_modeling_agent.barrier'
    result=agent.run_dir/'artifacts/result.json';result.parent.mkdir(exist_ok=True)
    result.write_text(json.dumps(dict(quantity='constrained_energy_gap',dE_act_kcal=10,
                                     validated_activation_free_energy=True)))
    review=agent.dispatch('inspect_barrier_result',dict(path=str(result)))
    assert not review['validated_activation_free_energy']
