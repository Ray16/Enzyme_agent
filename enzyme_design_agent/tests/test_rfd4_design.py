"""RFD4 validation, reservation boundary, input fidelity and output checks."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from enzyme_design_agent.bootstrap import design_tool_names, design_tool_schemas
from enzyme_design_agent.tools import rfd4_design as rfd
from enzyme_design_agent.tools import _rfd4_worker as worker


class RFD4ToolTests(unittest.TestCase):
    def test_registered_without_exposing_injected_directory(self):
        self.assertIn('check_rfd4_design_readiness', design_tool_names())
        self.assertIn('get_rfd4_design_status', design_tool_names())
        schema = next(s['function'] for s in design_tool_schemas()
                      if s['function']['name'] == 'design_enzyme_with_rfd4')
        self.assertNotIn('run_dir', schema['parameters']['properties'])
        self.assertEqual(schema['parameters']['required'], ['input_path'])

    def test_invalid_request_never_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'input.yaml'; source.write_text('example: {contig: "100"}')
            for options in ({'num_replicates':0}, {'num_replicates':True}, {'seed':-1},
                            {'recycle_schedule':'unknown'}):
                with mock.patch.object(rfd, '_preflight') as preflight, mock.patch.object(rfd.subprocess, 'Popen') as launch:
                    result = rfd.design_enzyme_with_rfd4(str(source), **options)
                    self.assertEqual(result['status'], 'error')
                    preflight.assert_not_called(); launch.assert_not_called()

    def test_remote_command_quotes_paths_and_leaves_gpu_visibility_to_gate(self):
        paths = dict(host='remote', hf_home='/cache/a b', python='/env/python', worker='/tools/worker.py')
        cmd = rfd._command(paths, '--run', '/runs/a b/request.json')
        self.assertEqual(cmd[:3], ['ssh','-o','BatchMode=yes'])
        self.assertIn("'/runs/a b/request.json'", cmd[-1])
        self.assertNotIn('CUDA_VISIBLE_DEVICES=', cmd[-1])
        self.assertIn('OMP_NUM_THREADS=2', cmd[-1])
        with self.assertRaises(ValueError):
            rfd._command(dict(paths, host='-oProxyCommand=bad'))

    def test_launch_is_persistent_and_uses_configured_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'input.cif'; source.write_text('data_test\n')
            with mock.patch.object(rfd, '_preflight', return_value={'ready':True,'free_gpus':['0']}), \
                 mock.patch.object(rfd.subprocess, 'Popen', return_value=mock.Mock(pid=os.getpid())) as launch:
                result = rfd.design_enzyme_with_rfd4(str(source), run_dir=tmp)
            self.assertEqual(result['status'], 'running')
            job = Path(result['data']['job_dir'])
            request = json.loads((job/'request.json').read_text())
            self.assertEqual(request['paths']['python'], rfd._paths()['python'])
            self.assertEqual(request['input_path'], str(source))
            self.assertTrue(launch.call_args.kwargs['start_new_session'])

    def test_spec_paths_preserve_conditions_and_resolve_all_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); (base/'out').mkdir()
            for name in ('target.cif','ligand.cif'):
                (base/name).write_text('data_test\n')
            spec = {'design': {'input':['target.cif','ligand.cif'], 'contig':'A1-10,100',
                               'conditions': {'C_SEQ': {'true':[{'select':"sel('A')"}]}}}}
            source = base/'input.json'; source.write_text(json.dumps(spec))
            target, count = worker._input(dict(input_path=str(source), output_dir=str(base/'out')))
            saved = json.loads(target.read_text())
            self.assertEqual(count, 1)
            self.assertEqual(saved['design']['input'], [str(base/'target.cif'),str(base/'ligand.cif')])
            self.assertEqual(saved['design']['conditions'], spec['design']['conditions'])
            self.assertEqual(json.loads(source.read_text()), spec)

    def test_worker_submits_through_gate_with_cpu_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); req = base/'request.json'
            req.write_text(json.dumps(dict(output_dir=tmp,paths=dict(gpu_reserve='/gate'))))
            def run(argv, **kwargs):
                worker._write(base/'result.json', {'status':'success'})
            with mock.patch.object(worker, 'preflight', return_value=dict(ready=True,free_gpus=['3'])), \
                 mock.patch.object(worker, '_cpus', return_value=[10,11]), \
                 mock.patch.object(worker.subprocess, 'run', side_effect=run) as launch:
                worker.run(req)
            argv = launch.call_args.args[0]
            self.assertEqual(argv[:9], ['/gate','run','3','--label','rfd4-design','--','taskset','-c','10,11'])
            self.assertIn('--generate', argv)

    def test_verification_rejects_missing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Expected 2 CIF designs, found 0'):
                worker._verify(Path(tmp), 2)

    def test_stale_process_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); req = base/'request.json'
            req.write_text(json.dumps(dict(output_dir=tmp)))
            (base/'state.json').write_text(json.dumps(dict(pid=os.getpid(),start_ticks='wrong')))
            (base/'run.log').write_text('stale worker')
            self.assertEqual(worker.status(req)['status'], 'error')


if __name__ == '__main__':
    unittest.main()
