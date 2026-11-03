"""Remove unqualified queued enzymes without interrupting useful legacy workers.

Already-running worker versions hold their task lists in memory. Explicit scope
exclusion leases prevent them from claiming the removed enzymes. These leases
are owned by this live CPU-only process, are not prediction/completion markers,
and are removed after the recorded legacy workers exit. Future workers read the
amended task list directly. Original manifests and amendment history are retained.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from core import digest, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    excluded=sorted({t['enzyme'] for t in manifest['tasks'] if t.get('good_start') is False})
    if not excluded:raise RuntimeError('No queued exclusions to apply')
    # Fail closed if an excluded prediction is already in progress.
    for path in (root/'claims').glob('*/owner.json'):
        if path.parent.name.split('_')[0] in excluded:
            raise RuntimeError('Excluded enzyme is already claimed: '+str(path))
    identity=dict(host=os.uname().nodename,pid=os.getpid(),
        process_start=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        cpu_affinity=sorted(os.sched_getaffinity(0)),kind='user_scope_exclusion',owns_gpu=False)
    leases=[]
    for e in excluded:
        lease=root/'enzyme_claims'/e
        lease.mkdir()  # Atomic race with any legacy worker: existing claim is a blocker.
        write_json(lease/'owner.json',dict(identity=identity,enzyme=e,state='excluded_by_user',
            reason='No qualifying starting geometry; do not submit BioEmu2.'))
        leases.append(lease)
    # Capture only the legacy GPU worker process identities, never unrelated jobs.
    legacy=[json.loads(p.read_text())['identity'] for p in (root/'workers').glob('*.json')]
    old_hash=digest(root/'manifest.json')
    archive=root/'manifest_history'/f'{old_hash}.json'
    write_json(archive,manifest)
    removed=[t for t in manifest['tasks'] if t['enzyme'] in excluded]
    manifest['tasks']=[t for t in manifest['tasks'] if t['enzyme'] not in excluded]
    manifest['excluded_enzymes']=excluded
    manifest['parent_manifest_sha256']=old_hash
    manifest['protocol']['submission_scope']='Only enzymes with at least one physically acceptable pose passing the geometry screen.'
    manifest['protocol']['limitations']=[x for x in manifest['protocol']['limitations'] if not x.startswith('Best-available starts')]
    manifest['protocol']['limitations'].append('Enzymes without qualifying starts are excluded from BioEmu2, not labeled experimentally inactive.')
    write_json(root/'manifest.json',manifest)
    amendment=dict(identity=identity,status='applied',excluded_enzymes=excluded,removed_tasks=removed,
        parent_manifest_sha256=old_hash,manifest_sha256=digest(root/'manifest.json'),
        legacy_workers=legacy,remaining_tasks=len(manifest['tasks']),applied_epoch=time.time(),
        reason='User requested BioEmu2 only for enzymes with at least one good geometry.')
    write_json(root/'scope_amendment.json',amendment)
    print('SCOPE_APPLIED',len(manifest['tasks']),'tasks; excluded',excluded,flush=True)
    # Keep explicit exclusion ownership live while old versions drain their work.
    by_host={i['host']:[x for x in legacy if x['host']==i['host']] for i in legacy}
    code='''import json,pathlib,sys
identities=json.load(sys.stdin);live=[]
for i in identities:
 try:
  if (pathlib.Path('/proc')/str(i['pid'])/'stat').read_text().split()[21]==i['process_start']:live.append(i['pid'])
 except FileNotFoundError:pass
print(json.dumps(live))
'''
    while True:
        active=[]
        for host,ids in by_host.items():
            try:
                output=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,
                    shlex.join(['python3','-c',code])],input=json.dumps(ids),text=True,capture_output=True,timeout=20)
                if output.returncode:raise RuntimeError(output.stderr)
                active.extend(json.loads(output.stdout))
            except (subprocess.SubprocessError,RuntimeError,ValueError):
                active.append('unresolved:'+host)
        if not active:break
        time.sleep(20)
    for lease in leases:
        record=json.loads((lease/'owner.json').read_text())
        if record['identity']!=identity:raise RuntimeError('Exclusion ownership changed')
        (lease/'owner.json').unlink();lease.rmdir()
    amendment.update(status='legacy_workers_drained',finished_epoch=time.time())
    write_json(root/'scope_amendment.json',amendment)
    print('EXCLUSION_LEASES_RELEASED',flush=True)


if __name__=='__main__':main()
