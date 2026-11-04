"""Offline GENzyme registration, reservation-boundary and output-validation tests."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from enzyme_design_agent.bootstrap import design_tool_names, design_tool_schemas
from enzyme_design_agent.tools import genzyme_design as gz
from enzyme_design_agent.tools._genzyme_worker import _microbatch


def write_protein(path, sequence='AAA'):
    serial=0; lines=[]
    for i in range(len(sequence)):
        for name,dx in [('N',0),('CA',1),('C',2)]:
            serial+=1
            lines.append(f'ATOM  {serial:5d} {name:^4s} ALA A{i+1:4d}    {i*3.8+dx:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{80.:6.2f}          {name[0]:>2s}  ')
    path.write_text('\n'.join(lines)+'\nEND\n')


class GenzymeToolsTest(unittest.TestCase):
    def test_memory_batching_keeps_all_inverse_folding_trials(self):
        calls=[]
        def sample(client,proteins,configs):
            calls.append((list(proteins),list(configs)))
            return [f'{p}:{c}' for p,c in zip(proteins,configs)]
        result=_microbatch(sample,1)('client',list(range(8)),list(range(10,18)))
        self.assertEqual(result,[f'{i}:{i+10}' for i in range(8)])
        self.assertEqual(len(calls),8)
        self.assertTrue(all(len(p)==len(c)==1 for p,c in calls))

    def test_registered_and_schema_excludes_injected_run_dir(self):
        names=design_tool_names()
        for name in ['check_genzyme_design_readiness','design_enzyme_with_genzyme','get_genzyme_design_status']:
            self.assertIn(name,names)
        schema=next(s['function'] for s in design_tool_schemas() if s['function']['name']=='design_enzyme_with_genzyme')
        self.assertNotIn('run_dir',schema['parameters']['properties'])
        self.assertEqual(schema['parameters']['required'],['substrate_smiles','product_smiles'])

    def test_invalid_dimensions_do_not_launch(self):
        with mock.patch.object(gz.subprocess,'Popen') as launch:
            result=gz.design_enzyme_with_genzyme('CCO','CC=O',n_protein_res=64,n_pocket_res=64)
        self.assertEqual(result['status'],'error');launch.assert_not_called()

    def test_gpu_discovery_filters_other_hosts_and_insufficient_memory(self):
        def run(argv,**kwargs):
            if argv[0]=='gate':return mock.Mock(stdout='host 0\nhost 1\nother 3\n',returncode=0)
            return mock.Mock(stdout='0, 16384\n1, 32768\n3, 95830\n',returncode=0)
        with mock.patch.object(gz.socket,'gethostname',return_value='host'),mock.patch.object(gz.subprocess,'run',side_effect=run):
            self.assertEqual(gz._free_gpus({'gpu_reserve':'gate'}),['1'])

    def test_no_gpu_retains_request_without_starting_job(self):
        paths=dict(repo='/repo',python='/python',checkpoint='/weights',worker='/worker',gpu_reserve='/gate',hf_home='/hf')
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(gz,'_readiness',return_value=dict(ready=True,paths=paths)), \
             mock.patch.object(gz,'_cpu_slots',return_value=[10,11]), \
             mock.patch.object(gz,'_free_gpus',return_value=[]), \
             mock.patch.object(gz.subprocess,'run'),mock.patch.object(gz.subprocess,'Popen') as popen:
            result=gz.design_enzyme_with_genzyme('CCO','CC=O',run_dir=tmp)
            self.assertEqual(result['status'],'error')
            self.assertTrue((Path(result['data']['job_dir'])/'request.json').exists())
            popen.assert_not_called()

    def test_launch_uses_gate_cpu_affinity_and_dedicated_worker(self):
        paths=dict(repo='/repo',python='/genzyme/python',checkpoint='/weights',worker='/tools/_genzyme_worker.py',gpu_reserve='/gate',hf_home='/hf')
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(gz,'_readiness',return_value=dict(ready=True,paths=paths)), \
             mock.patch.object(gz,'_cpu_slots',return_value=[10,11]), \
             mock.patch.object(gz,'_free_gpus',return_value=['3']), \
             mock.patch.object(gz.subprocess,'run') as preflight, \
             mock.patch.object(gz.subprocess,'Popen',return_value=mock.Mock(pid=os.getpid())) as popen:
            result=gz.design_enzyme_with_genzyme('CCO','CC=O',run_dir=tmp)
            self.assertEqual(result['status'],'running')
            argv=popen.call_args.args[0]
            self.assertEqual(argv[:6],['/gate','run','3','--label','genzyme-design','--'])
            self.assertEqual(argv[6:10],['taskset','-c','10,11','/genzyme/python'])
            self.assertIn('/tools/_genzyme_worker.py',argv)
            self.assertIn('--validate-only',preflight.call_args.args[0])
            env=popen.call_args.kwargs['env']
            for name in gz._THREAD_KEYS:self.assertEqual(env[name],'2')
            self.assertEqual(popen.call_args.kwargs['start_new_session'],True)
            request=json.loads((Path(result['data']['job_dir'])/'request.json').read_text())
            self.assertEqual(request['substrate_smiles'],'CCO')
            self.assertEqual(request['max_attempts'],3)


class GenzymeStatusTest(unittest.TestCase):
    def prepare(self,root):
        (root/'request.json').write_text(json.dumps({'n_protein_res':3,'n_pocket_res':3,'num_designs':1}))
        (root/'design').mkdir()
        (root/'design/sequence.fasta').write_text('>design\nAAA\n')
        write_protein(root/'design/protein.pdb')
        write_protein(root/'design/pocket.pdb')
        result={'designs':[dict(sequence='AAA',fasta='design/sequence.fasta',protein_pdb='design/protein.pdb',pocket_pdb='design/pocket.pdb')]}
        (root/'result.json').write_text(json.dumps(result))

    def test_completed_outputs_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.prepare(Path(tmp))
            self.assertEqual(gz.get_genzyme_design_status(tmp)['status'],'success')

    def test_marker_does_not_override_missing_or_wrong_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.prepare(root)
            (root/'design/sequence.fasta').write_text('>design\nACA\n')
            self.assertEqual(gz.get_genzyme_design_status(tmp)['status'],'error')
            (root/'design/sequence.fasta').write_text('>design\nAAA\n')
            (root/'design/protein.pdb').write_text('wrong\n'*100)
            self.assertEqual(gz.get_genzyme_design_status(tmp)['status'],'error')

    def test_nonfinite_structure_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.prepare(root)
            f=root/'design/protein.pdb';lines=f.read_text().splitlines()
            lines[0]=lines[0][:30]+'     nan'+lines[0][38:]
            f.write_text('\n'.join(lines)+'\n')
            self.assertEqual(gz.get_genzyme_design_status(tmp)['status'],'error')

    def test_exited_launcher_without_outputs_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'request.json').write_text('{}')
            (root/'launch.json').write_text(json.dumps(dict(host=gz.socket.gethostname().split('.')[0],pid=os.getpid(),start_ticks='wrong-start-time')))
            (root/'run.log').write_text('gpu_reserve BUSY/reserved')
            result=gz.get_genzyme_design_status(tmp)
            self.assertEqual(result['status'],'error')
            self.assertIn('BUSY',result['data']['log_tail'])

    def test_remote_liveness_is_not_assumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'request.json').write_text('{}')
            (root/'launch.json').write_text(json.dumps(dict(host='unreachable-other-host',pid=123,start_ticks='1')))
            self.assertEqual(gz.get_genzyme_design_status(tmp)['status'],'unknown')


if __name__=='__main__':unittest.main()
