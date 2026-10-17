"""Check release identity, embedded weights and reported completed-game counts."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]

def main():
    release=json.loads((ROOT/'evidence/release.json').read_text())
    agent=(ROOT/'submission/agent.py').read_bytes()
    digest=hashlib.sha256(agent).hexdigest()
    assert digest==release['agent_sha256']
    for run in sorted((ROOT/'evidence').iterdir()):
        if not run.is_dir():continue
        summary=json.loads((run/'summary.json').read_text())
        manifest=json.loads((run/'manifest.json').read_text())
        games=[json.loads(line) for line in (run/'games.jsonl').read_text().splitlines()]
        assert summary['complete'] and len(games)==manifest['games']==summary['a']['games'],run.name
        assert len({g['game'] for g in games})==len(games)
        assert sum(g['winner']=='a' for g in games)==summary['a']['wins']
        assert sum(g['winner']=='b' for g in games)==summary['b']['wins']
        for side in ('a','b'):
            filename=manifest[side].replace('\\','/').rsplit('/',1)[-1]
            source=run/side/filename
            assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_hashes'][side]
        print(f"{run.name}: {summary['a']['wins']}/{len(games)} verified")
    tree=ast.parse(agent)
    values={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in ('ML_BIAS','ML_TREES'):values[node.targets[0].id]=ast.literal_eval(node.value)
    model=json.loads((ROOT/'models/value_model.json').read_text())
    assert values['ML_BIAS']==model['bias'] and values['ML_TREES']==model['trees']
    assert len(model['trees'])==64
    with zipfile.ZipFile(ROOT/'releases/Odyssey_submission.zip') as z:
        assert z.namelist()==['agent.py'] and z.read('agent.py')==agent
        assert z.testzip() is None
    print('Release hash, embedded model, archive and all selected evidence verified.')

if __name__=='__main__':main()
