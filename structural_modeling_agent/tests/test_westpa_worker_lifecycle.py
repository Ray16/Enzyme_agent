import os
from pathlib import Path
import subprocess
import sys


def test_worker_supervisor_reaps_orphaned_grandchildren(tmp_path):
    # A fresh supervisor process avoids changing pytest's child-reaper policy.
    script = r'''
import os,sys,subprocess,time
from pathlib import Path
import psutil
from structural_modeling_agent.sampling.westpa_gpu_worker import enable_subreaper,cleanup_descendants
enable_subreaper()
code="import os,time; pid=os.fork(); time.sleep(30) if pid==0 else None"
p=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
p.wait(timeout=5)
children=psutil.Process().children(recursive=True)
assert len(children)==1,children
pid=children[0].pid
assert children[0].ppid()==os.getpid()
assert pid in cleanup_descendants()
assert not psutil.Process().children(recursive=True)
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    subprocess.run([sys.executable, "-c", script], env=env, timeout=15, check=True)
