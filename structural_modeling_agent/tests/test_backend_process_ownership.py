"""A completed producer and stale driver observations do not cause false conflicts."""
from structural_modeling_agent.lambda_boltz_backend import unexpected_residents


class Process:
    pid=100
    def __init__(self,code):self.code=code
    def poll(self):return self.code


def test_postexit_observation_does_not_stop_a_completed_prediction():
    assert unexpected_residents(Process(0),{200},set(),{200},{200:200})==set()


def test_reparented_worker_in_the_owned_session_is_still_ours():
    assert unexpected_residents(Process(None),{200},set(),{200},{200:100})==set()


def test_live_foreign_process_is_a_conflict_but_stale_driver_pid_is_not():
    assert unexpected_residents(Process(None),{200,300},{200},{200,300},{200:100,300:300})=={300}
    assert unexpected_residents(Process(None),{200,300},{200},{200},{200:100})==set()
