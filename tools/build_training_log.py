"""Build an auditable log and reproduce the frozen fit without changing the release."""
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'training_logs'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def export_model(model):
    return {'bias':float(model._baseline_prediction[0,0]),
            'trees':[[[int(n['feature_idx']),float(n['num_threshold']),int(n['left']),
                       int(n['right']),float(n['value']),int(n['is_leaf'])]
                      for n in stage[0].nodes] for stage in model._predictors]}

def main():
    import numpy as np
    import scipy
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score,log_loss
    from threadpoolctl import threadpool_limits
    OUT.mkdir(exist_ok=True)
    started=datetime.now(timezone.utc).isoformat()
    dataset=ROOT/'training/value_training.npz'
    saved=np.load(dataset)
    X,y,games=saved['X'],saved['y'],saved['game']
    test=games%5==0
    original=json.loads((ROOT/'models/training_metrics.json').read_text())
    release=json.loads((ROOT/'evidence/release.json').read_text())
    exported=json.loads((ROOT/'models/value_model.json').read_text())
    assert X.shape==(original['states'],original['features'])
    assert len(y)==len(games)==len(X) and np.isfinite(X).all()
    assert set(np.unique(y))=={0,1}
    assert set(np.unique(games[test])).isdisjoint(set(np.unique(games[~test])))
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [
        dataset,ROOT/'training/train_value.py',ROOT/'training/heuristic_agent.py',
        ROOT/'training/learned_features.py',ROOT/'models/value_model.json',
        ROOT/'models/training_metrics.json',ROOT/'submission/agent.py']}
    assert hashes['submission/agent.py']==release['agent_sha256']
    data_info={
        'positions':len(X),'features':X.shape[1],'games':len(np.unique(games)),
        'game_id_min':int(games.min()),'game_id_max':int(games.max()),
        'fit_games':len(np.unique(games[~test])),'heldout_games':len(np.unique(games[test])),
        'fit_positions':int((~test).sum()),'heldout_positions':int(test.sum()),
        'positive_labels':int(y.sum()),'negative_labels':int((y==0).sum()),
        'split_rule':'Whole game held out if game_id % 5 == 0',
        'labels':'Eventual win for the player perspective; positions collected at END_TURN',
        'input_dtype':str(X.dtype),'finite_features':True,'game_overlap':0}
    (OUT/'dataset_summary.json').write_text(json.dumps(data_info,indent=2))
    (OUT/'original_training_metrics.json').write_bytes((ROOT/'models/training_metrics.json').read_bytes())
    # These are exact counts recovered from the saved dataset, not historical stdout.
    with (OUT/'reconstructed_collection_counts.jsonl').open('w',encoding='utf-8') as f:
        for end in range(20,1001,20):
            record={'kind':'reconstructed_from_saved_dataset','games_through':end,
                    'positions_in_batch':int(((games>=end-20)&(games<end)).sum()),
                    'positions_cumulative':int((games<end).sum())}
            f.write(json.dumps(record)+'\n')
    params=dict(max_iter=64,max_leaf_nodes=7,max_depth=3,learning_rate=.1,
                l2_regularization=15,min_samples_leaf=80,random_state=483)
    model=HistGradientBoostingClassifier(**params,verbose=1)
    t0=time.perf_counter()
    raw=io.StringIO()
    with contextlib.redirect_stdout(raw),threadpool_limits(limits=2):
        model.fit(X[~test],y[~test])
        predicted=model.predict_proba(X[test])[:,1]
    elapsed=time.perf_counter()-t0
    actual=export_model(model)
    match=actual==exported
    fresh_auc=float(roc_auc_score(y[test],predicted))
    fresh_loss=float(log_loss(y[test],predicted))
    original_model_match=match
    def infer(x):
        total=exported['bias']
        for tree in exported['trees']:
            n=0
            while not tree[n][5]:
                node=tree[n];n=node[2] if x[node[0]]<=node[1] else node[3]
            total+=tree[n][4]
        return total
    with threadpool_limits(limits=2):
        delta=float(np.max(np.abs(np.array([infer(x) for x in X[test][:100]])-model.decision_function(X[test][:100]))))
    reproduction={'kind':'new_reproduction_run_from_frozen_dataset',
        'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),
        'wall_seconds':elapsed,'boosting_iterations':int(model.n_iter_),
        'heldout_game_auc':fresh_auc,'heldout_logloss':fresh_loss,
        'exported_trees_exactly_match_release':original_model_match,
        'pure_python_vs_fitted_max_logit_error_100_vectors':delta,
        'auc_matches_recorded_metric':abs(fresh_auc-original['heldout_game_auc'])<1e-12,
        'logloss_matches_recorded_metric':abs(fresh_loss-original['heldout_logloss'])<1e-12}
    (OUT/'refit_stdout.log').write_text(
        'NEW REPRODUCTION RUN. This is a real fit performed when generating the training-log supplement.\n'
        'It is not the original training console transcript.\nStarted UTC: '+started+'\n\n'
        +raw.getvalue()+'\n'+json.dumps(reproduction,indent=2)+'\n',encoding='utf-8')
    with (OUT/'refit_iterations.jsonl').open('w',encoding='utf-8') as f:
        for i in range(len(model.train_score_)):
            record={'kind':'new_reproduction_fit','iteration':i,
                    'training_negative_log_loss':float(model.train_score_[i]),
                    'internal_validation_negative_log_loss':float(model.validation_score_[i])}
            f.write(json.dumps(record)+'\n')
    env={'python':platform.python_version(),'platform':platform.platform(),
         'numpy':np.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,
         'fit_thread_limit':2}
    manifest={'log_type':'Evidence reconstruction plus a separately timestamped deterministic refit',
        'original_console_transcript':'Not available as a saved standalone file in this snapshot',
        'release_sha256':release['agent_sha256'],'input_hashes':hashes,
        'environment_of_reproduction':env,'dataset':data_info,'parameters':model.get_params(),
        'original_recorded_metrics':original,'reproduction':reproduction,
        'collection_seed_rules':{'board_seed':'1500000 + game_id','dice_seed':'6000001 + game_id',
                                'agent_rng_seed':'90001 + 2 * game_id + player_index'},
        'offline_clock_observation':.8,
        'clock_disclosure':'Data generation uses an offline driver, not an official-clock score. Gameplay benchmarks use the unmodified official match harness.',
        'source_commit_before_log':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT/'reproduction_result.json').write_text(json.dumps(reproduction,indent=2),encoding='utf-8')
    rows=[]
    for name in ['learned_control','learned_04','learned_09','learned_final_1000']:
        run=ROOT/'evidence'/name
        r=json.loads((run/'summary.json').read_text())
        m=json.loads((run/'manifest.json').read_text())
        a=r['a']
        rows.append(f"| [{name}](evidence/{name}/summary.json) | {a['wins']}/{a['games']} | {a['win_rate']*100:.2f}% | {a['mean_cpu_s']*1000:.1f} | {a['errors']} | {a['games_over_budget']} |")
    text=f"""# Odyssey training log

Team Odyssey. Supplement for the frozen learned release.

**Status:** This log reconstructs the original training process from preserved source, data and metrics. It also includes a new, timestamped refit of the saved dataset. Reconstructed counts and new refit output are labeled separately. No historical timestamps or per-iteration measurements have been invented.

## Release identity

- Runtime source: [submission/agent.py](submission/agent.py).
- Agent SHA-256: {release['agent_sha256']}
- Model export: [models/value_model.json](models/value_model.json).
- Fitted weights are embedded in the runtime as ML_BIAS and ML_TREES.
- The original standalone training console transcript is unavailable in this snapshot. The original summary metrics, dataset and model are preserved.

## Original data generation

The heuristic teacher played F in 1,000 offline games, alternating seats. The trainer collected end-of-turn states from both player perspectives and labeled each state with the eventual winner. Each player's random state was isolated from the simulated dice stream. Collection is an offline learning process with a constant 0.8-second clock observation, not a competition benchmark.

| Item | Preserved value |
|---|---:|
| Games represented | {data_info['games']} |
| Labeled positions | {data_info['positions']:,} |
| Features | {data_info['features']} |
| Fit games / held-out games | {data_info['fit_games']} / {data_info['heldout_games']} |
| Fit positions / held-out positions | {data_info['fit_positions']:,} / {data_info['heldout_positions']:,} |
| Positive / negative labels | {data_info['positive_labels']:,} / {data_info['negative_labels']:,} |
| Game overlap across the split | 0 |

Board seed = 1500000 + game ID. Dice seed = 6000001 + game ID. Agent random-state seed = 90001 + 2 × game ID + player index. Whole games with game ID divisible by five form the external validation split. The [collection-count log](training_logs/reconstructed_collection_counts.jsonl) recovers exact 20-game batch counts from the saved dataset. It does not reconstruct historical elapsed times.

Features describe information available to the player: permitted points, production, resource hands, ports, robber effects, buildings, roads, played knights and build affordability. Opponent unplayed development-card types are excluded.

## Original fitting configuration

| Parameter | Value |
|---|---|
| Estimator | HistGradientBoostingClassifier |
| Maximum boosting iterations | 64 |
| Maximum depth / leaves per tree | 3 / 7 |
| Learning rate | 0.1 |
| L2 regularization | 15 |
| Minimum samples per leaf | 80 |
| Random state | 483 |
| Internal validation fraction | 0.1, scikit-learn default |
| Internal early stopping | auto, scikit-learn default |
| Numeric input | float64 |

The internal validation split is drawn from the fit subset. The separate held-out game subset never enters model fitting. Full resolved estimator settings appear in [training_logs/manifest.json](training_logs/manifest.json).

## Preserved original metrics

- Held-out-game AUC: **{original['heldout_game_auc']:.10f}**.
- Held-out log loss: **{original['heldout_logloss']:.10f}**.
- Recorded end-to-end collection and fitting wall time: **{original['wall_s']:.6f} seconds**.
- Runtime model: **64 trees**. Exported pure-Python inference was checked against the fitted model.

The original timing includes collection and fitting. It is not a per-epoch timing or a gameplay CPU measurement. AUC describes outcome prediction, not win rate.

## Newly executed reproduction

- Started: {reproduction['started_utc']}.
- Finished: {reproduction['finished_utc']}.
- Refit wall time: {elapsed:.6f} seconds.
- Trees exactly match the preserved release model: **{match}**.
- Refit AUC: {fresh_auc:.10f}.
- Refit log loss: {fresh_loss:.10f}.
- Maximum pure-Python logit error on 100 held-out vectors: {delta:.12g}.

The [actual refit console log](training_logs/refit_stdout.log), [iteration scores](training_logs/refit_iterations.jsonl) and [machine-readable result](training_logs/reproduction_result.json) belong to this new verification run. Iteration 0 is the pre-boosting score. The iteration log reports negative log loss on the fit and internal validation subsets. It must not be presented as a recovered historical training transcript.

## Policy selection and gameplay validation

| Run | Wins / games | Vs F | Mean local CPU ms | Errors | Over-clock |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The first three runs share the same 400-game screen. The highest-win clean candidate, the 0.4 model blend, was frozen before the fresh 1,000-game test. Its two-win screening lead is weak evidence of improvement. The final 79.7% is a local official-harness result, not an organizer leaderboard score. O and group robustness are unmeasured.

## Reproduce and inspect

After setting CATAN_KIT_ROOT and using the pinned Python 3.11 environment:

~~~bash
python tools/verify_evidence.py
python tools/build_training_log.py
~~~

The log builder refits the saved dataset and writes only the log supplement. It does not replace the packaged agent, the model export or the saved dataset. [manifest.json](training_logs/manifest.json) records input SHA-256 hashes, software versions and all settings. The report, model, runtime ZIP and benchmark evidence remain tied to the same source hash.
"""
    (ROOT/'TRAINING_LOG.md').write_text(text,encoding='utf-8')
    (ROOT/'TRAINING_LOG.txt').write_text(text,encoding='utf-8')
    print(json.dumps({'log':str(ROOT/'TRAINING_LOG.md'),'dataset':data_info,'reproduction':reproduction},indent=2))
    assert match and reproduction['auc_matches_recorded_metric'] and reproduction['logloss_matches_recorded_metric']
    assert delta<1e-9
    assert hashes=={name:sha(ROOT/name) for name in hashes},'A frozen input changed'

if __name__=='__main__':main()
