# Learned value parameters

value_model.json contains the 64-tree model in the verified release. Each node stores:

~~~text
[feature_index, threshold, left_child, right_child, value, is_leaf]
~~~

The bias initializes the logit. Each tree contributes its reached leaf value. Identical parameters appear in submission/agent.py as ML_BIAS and ML_TREES.

The model uses training/learned_features.py, with 91 features. Outside opening placement, the agent adds 0.4 times the raw logit to the heuristic. training_metrics.json records the original fit.

tools/verify_evidence.py compares the JSON export with the embedded parameters. Older .pt files in my_team/ are unrelated historical experiments.
