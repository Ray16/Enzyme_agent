import json
import socket

import yaml

from structural_modeling_agent import family_inflight as guard


def job(root,identifier='same',smiles='CCO'):
    root.mkdir(exist_ok=True)
    path=root/(identifier+'.yaml')
    path.write_text(yaml.safe_dump({'sequences':[{'protein':{'id':'A','sequence':'AGGA'}},{'ligand':{'id':'B','smiles':smiles}}]}))
    return {'id':identifier,'input_yaml':str(path)}


def test_identical_systems_in_different_runs_have_only_one_owner(tmp_path):
    first=tmp_path/'first';second=tmp_path/'second';a=job(first);b=job(second)
    claim=guard.acquire(first,a,'one');assert claim is not None
    assert guard.acquire(second,b,'two') is None
    guard.release(claim)
    claim=guard.acquire(second,b,'two');assert claim is not None
    guard.release(claim)


def test_different_molecules_do_not_share_a_claim(tmp_path):
    first=tmp_path/'first';second=tmp_path/'second'
    a=guard.acquire(first,job(first),'one');b=guard.acquire(second,job(second,smiles='CCC'),'two')
    assert a!=b
    guard.release(a);guard.release(b)


def test_legacy_native_job_blocks_duplicate_after_frontend_exits(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';a=job(old);b=job(new)
    claim=old/'family_cofold/claims/same';claim.mkdir(parents=True)
    (claim/'owner.json').write_text(json.dumps({'host':socket.gethostname(),'pid':999999999,'process_start_ticks':'missing'}))
    (old/'family_cofold/queue.json').write_text(json.dumps({'items':[a]}))
    native=old/'family_cofold/workers/one/provider/backend_jobs/same';native.mkdir(parents=True)
    (native/'receipt.json').write_text(json.dumps({'status':'running'}))
    assert guard.acquire(new,b,'new') is None
    (native/'receipt.json').write_text(json.dumps({'status':'failed'}))
    result=guard.acquire(new,b,'new');assert result is not None
    guard.release(result)


def test_uncertain_remote_ownership_is_not_reclaimed(tmp_path,monkeypatch):
    first=tmp_path/'first';second=tmp_path/'second';a=job(first);b=job(second)
    claim=guard.acquire(first,a,'first')
    record=json.loads((claim/'owner.json').read_text());record.update(host='unreachable-host',pid=123)
    (claim/'owner.json').write_text(json.dumps(record))
    monkeypatch.setattr(guard,'_live',lambda owner:None)
    assert guard.acquire(second,b,'second') is None


def test_orphan_native_job_without_a_local_claim_still_blocks_other_runs(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';a=job(old);b=job(new)
    native=old/'family_cofold/workers/one/provider/backend_jobs/same';native.mkdir(parents=True)
    (native/'request.json').write_text(json.dumps(a))
    (native/'receipt.json').write_text(json.dumps({'status':'running'}))
    assert guard.acquire(new,b,'new') is None
    own=guard.acquire(old,a,'reattach');assert own is not None
    guard.release(own)


def test_claim_is_not_visible_before_its_owner_is_written(tmp_path,monkeypatch):
    first=tmp_path/'first';candidate=job(first)
    published=[];write=guard.atomic_write_json
    def observe(path,value):
        target=tmp_path/'.family_inflight'/value['physical_system_key']
        assert not target.exists()
        published.append(target)
        return write(path,value)
    monkeypatch.setattr(guard,'atomic_write_json',observe)
    claim=guard.acquire(first,candidate,'one')
    assert published==[claim] and (claim/'owner.json').exists()
    guard.release(claim)


def test_release_retires_owner_and_directory_together(tmp_path):
    first=tmp_path/'first';claim=guard.acquire(first,job(first),'one')
    record=json.loads((claim/'owner.json').read_text())
    guard.release(claim)
    assert not claim.exists()
    archived=list((tmp_path/'.family_inflight/archive').glob('*_released_*/owner.json'))
    assert len(archived)==1 and json.loads(archived[0].read_text())==record
