"""Exercise deferred submission timing without remote calls or allocations."""
import importlib.util
from pathlib import Path
import tempfile
from unittest.mock import Mock
import unittest

CAMPAIGN=Path(__file__).resolve().parents[1]/'campaigns/amide_AS9_AS17_AS23_step1_2Mg_saturation_5A_20261019'


def module(name):
    spec=importlib.util.spec_from_file_location(name,CAMPAIGN/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


watch=module('aurora_resume_watch')


class Clock:
    value=0
    def __call__(self):return self.value
    def sleep(self,seconds):
        assert seconds>0
        self.value+=seconds


class ResumeWatchTests(unittest.TestCase):
    def setup_run(self):
        self.clock=Clock();self.events=[]
        self.schedule=dict(first_probe=10,interval_seconds=10,deadline=27)
        self.kw=dict(clock=self.clock,sleep=self.clock.sleep,
                     probe=Mock(return_value={'ready':False}),ready=Mock(return_value=True),
                     launch=Mock(),submitted=Mock(return_value=[]),
                     save=lambda stage,**kw:self.events.append((stage,kw)))

    def test_first_probe_is_deferred_then_hourly_until_absolute_deadline(self):
        self.setup_run();times=[]
        self.kw['probe']=lambda:times.append(self.clock()) or {'ready':False}
        self.assertEqual(watch.run_watch(self.schedule,**self.kw),'expired')
        self.assertEqual(times,[10,20]);self.assertEqual(self.clock(),27)
        self.kw['launch'].assert_not_called()

    def test_ready_aurora_still_requires_msas(self):
        self.setup_run();self.kw['probe'].return_value={'ready':True};self.kw['ready'].return_value=False
        watch.run_watch(self.schedule,**self.kw)
        self.kw['launch'].assert_not_called()

    def test_successful_submission_stops_further_probes(self):
        self.setup_run();self.kw['probe'].return_value={'ready':True}
        jobs=[{'job_id':'123.pbs'},{'job_id':'124.pbs'}]
        self.kw['launch'].side_effect=lambda s:self.kw['submitted'].configure_mock(return_value=jobs)
        self.assertEqual(watch.run_watch(self.schedule,**self.kw),'submitted')
        self.kw['probe'].assert_called_once();self.kw['launch'].assert_called_once()
        self.assertEqual(self.clock(),10)

    def test_probe_failure_retries_only_on_next_hour(self):
        self.setup_run();times=[]
        def fail():
            times.append(self.clock());raise ConnectionError('maintenance')
        self.kw['probe']=fail
        watch.run_watch(self.schedule,**self.kw)
        self.assertEqual(times,[10,20]);self.kw['launch'].assert_not_called()

    def test_probe_returning_after_deadline_cannot_launch(self):
        self.setup_run()
        def slow():
            self.clock.value=30;return {'ready':True}
        self.kw['probe']=slow
        self.assertEqual(watch.run_watch(self.schedule,**self.kw),'expired')
        self.kw['launch'].assert_not_called()

    def test_expired_schedule_does_not_probe_or_launch(self):
        self.setup_run();self.clock.value=28
        watch.run_watch(self.schedule,**self.kw)
        self.kw['probe'].assert_not_called();self.kw['launch'].assert_not_called()

    def test_aurora_rejects_expired_submission_before_reading_inputs(self):
        control=module('control_ops')
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError,'window expired'):
                control.submit_phase(root,'single_mutants',[16,16],'12:00:00',{},not_after=0)
            self.assertFalse(list(Path(root).iterdir()))


if __name__=='__main__':unittest.main()
