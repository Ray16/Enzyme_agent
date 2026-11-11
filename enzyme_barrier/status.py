"""Read-only campaign status: execution completion is separate from validation."""
import argparse
import json
from pathlib import Path


def status(root):
    root=Path(root);campaign=json.loads((root/'campaign.json').read_text());tasks=[]
    for name in campaign['worker_manifests']:
        path=root/name
        if not path.exists():continue
        cfg=json.loads(path.read_text())
        for task in cfg['tasks']:
            state=root/'tasks'/task['id']/'state.json'
            d=json.loads(state.read_text()) if state.exists() else {}
            tasks.append(dict(id=task['id'],device=cfg.get('device','gpu'),gpu=cfg.get('gpu'),status=d.get('status','queued'),attempts=d.get('attempts',0)))
    return dict(campaign_id=campaign['campaign_id'],tasks=tasks,scientific_scope=campaign['systems'],
                validated_barriers=campaign.get('validated_barriers',[]),limitations=campaign.get('limitations',[]))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');a=p.parse_args(argv);print(json.dumps(status(a.run),indent=2))


if __name__=='__main__':main()
