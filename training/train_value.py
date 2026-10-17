"""Offline training only. Final competition benchmarks use the original harness."""
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ['CATAN_KIT_ROOT']).resolve()
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE))

def load_agent(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent

def collect_batch(job):
    from catanatron.game import Game
    from catanatron.models.player import RandomPlayer,Color
    from catanatron.models.enums import RESOURCES,ActionType
    from catan_kit.obs import Obs
    from learned_features import public_features
    start,count=job
    strong=load_agent(HERE/'heuristic_agent.py','strong')
    opponent=load_agent(ROOT/'examples/f_agent.py','opponent')
    records=[]
    for number in range(start,start+count):
        colors=[Color.RED,Color.BLUE]
        agents=[strong(),opponent()] if number%2==0 else [opponent(),strong()]
        states=[random.Random(90001+number*2+i).getstate() for i in range(2)]
        g=Game([RandomPlayer(c) for c in colors],seed=1500000+number)
        random.seed(6000001+number)
        node_prod={}
        for node,tiles in g.state.board.map.adjacent_tiles.items():
            pp=[0.]*5
            for t in tiles:
                if t.resource: pp[RESOURCES.index(t.resource)]+=(6-abs(7-t.number))/36
            node_prod[node]=pp
        samples=[]
        last={}
        def decide(player,world,actions):
            idx=colors.index(player.color)
            obs=Obs(world,player.color)
            obs.cpu_left=.8
            # Per-player private RNG state must not change the simulated dice stream.
            engine_rng=random.getstate()
            random.setstate(states[idx])
            choice=int(agents[idx].decide(obs,actions))
            states[idx]=random.getstate()
            random.setstate(engine_rng)
            action=actions[choice]
            turn=world.state.num_turns
            if not world.state.is_initial_build_phase and action.action_type==ActionType.END_TURN and last.get(idx)!=turn:
                for c in colors:
                    samples.append((public_features(world,c,node_prod),c))
                last[idx]=turn
            return action
        winner=g.play(decide_fn=decide)
        if winner is not None:
            records.extend((number,x,int(winner==c)) for x,c in samples)
    return records

def export_model(model):
    trees=[]
    for stage in model._predictors:
        nodes=stage[0].nodes
        trees.append([[int(n['feature_idx']),float(n['num_threshold']),int(n['left']),int(n['right']),float(n['value']),int(n['is_leaf'])] for n in nodes])
    return {'bias':float(model._baseline_prediction[0,0]),'trees':trees}

def main():
    from catan_kit._patch import ensure_hash_seed
    ensure_hash_seed()
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--games',type=int,default=1000)
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    start=time.time()
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[(i,min(20,args.games-i)) for i in range(0,args.games,20)]
        for i,part in enumerate(pool.map(collect_batch,jobs)):
            rows.extend(part)
            print(json.dumps({'training_games':min((i+1)*20,args.games),'states':len(rows),'wall_s':round(time.time()-start,1)}),flush=True)
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score,log_loss
    from threadpoolctl import threadpool_limits
    data=np.array([r[1] for r in rows],dtype=np.float64)
    target=np.array([r[2] for r in rows])
    test=np.array([r[0]%5==0 for r in rows])
    np.savez_compressed(HERE/'value_training.npz',X=data,y=target,game=np.array([r[0] for r in rows]))
    with threadpool_limits(limits=2):
        model=HistGradientBoostingClassifier(max_iter=64,max_leaf_nodes=7,max_depth=3,learning_rate=.1,l2_regularization=15,min_samples_leaf=80,random_state=483)
        model.fit(data[~test],target[~test])
        predicted=model.predict_proba(data[test])[:,1]
        metrics={'training_games':args.games,'states':len(rows),'features':data.shape[1],'heldout_game_auc':roc_auc_score(target[test],predicted),'heldout_logloss':log_loss(target[test],predicted),'wall_s':time.time()-start}
        print(json.dumps(metrics),flush=True)
        (HERE/'value_training_metrics.json').write_text(json.dumps(metrics,indent=2))
        # Keep the held-out games out of fitting; gameplay benchmarks are separate.
        exported=export_model(model)
        (HERE/'value_model.json').write_text(json.dumps(exported,separators=(',',':')))
        sample=data[test][:100]
        def raw(x):
            score=exported['bias']
            for tree in exported['trees']:
                i=0
                while not tree[i][5]:
                    n=tree[i]; i=n[2] if x[n[0]]<=n[1] else n[3]
                score+=tree[i][4]
            return score
        error=float(np.max(np.abs(np.array([raw(x) for x in sample])-model.decision_function(sample))))
        assert error<1e-9,error
        print('EXPORTED_MODEL_MATCHES',error,flush=True)

if __name__=='__main__':main()
