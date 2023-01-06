import os
import json
try:
    PROJECT_PATH = '/home/ekernf01/Desktop/jhu/research/projects/perturbation_prediction/cell_type_knowledge_transfer/'
    os.chdir(os.path.join(PROJECT_PATH, "ggrn_backend3"))
except FileNotFoundError:
    PROJECT_PATH = 'gary_path'
    os.chdir(os.path.join(PROJECT_PATH, "ggrn_backend3"))

import sys 
import unittest
sys.path.append(os.path.expanduser(os.path.join(PROJECT_PATH, 'ggrn_backend3', 'src'))) 
sys.path.append(os.path.expanduser(os.path.join(PROJECT_PATH, 'network_collection', 'load_networks'))) 
import ggrn_backend3
from itertools import product
import pandas as pd
import numpy as np
import anndata 
from timeit import default_timer as timer

def simulate_autoregressive(
    num_controls = 2, 
    num_features = 5, 
    seed = 0, 
    num_steps = 1, 
    embed_in_higher_dimension = False
):
    # This is a pretty normal simulation combining controls and perturbed samples, with linear propagation of perturbation effects.
    # The weirdest thing here is we mark some samples as "treatment", even though no gene is perturbed.
    # Even tho there is no treatment, they may differ because the controls are not initialized at steady state, 
    # and because they are marked as treatment, the below code will advance them forward in time.
    metadata = pd.DataFrame({
        "matched_control":                      np.tile([i for i in range(num_controls)], num_features + 2),
        "is_control":                           np.repeat([True,  False], [num_controls, (num_features+1)*num_controls]),
        "is_treatment":                         np.repeat([False, True ], [num_controls, (num_features+1)*num_controls]),
        "is_steady_state":                      [False]*(num_features+2)*num_controls,
        "perturbation":                         np.repeat(["-999", "-999"] + [str(i) for i in range(num_features) ], num_controls),
        "expression_level_after_perturbation":  np.repeat(10, (num_features + 2)*num_controls),
    })
    metadata.index = [str(i) for i in metadata.index]
    np.random.seed(seed)
    all_controls = np.random.random((num_controls, num_features))
    G = np.random.random((num_features,num_features)) + np.eye(num_features)

    def perturb(one_control, perturbation, expression_level_after_perturbation):
        x = one_control.copy()
        if int(perturbation) >= 0:
            x[int(perturbation)] = float(expression_level_after_perturbation)
        return x
    def advance(one_control, perturbation, expression_level_after_perturbation, num_steps):
        one_control = perturb(one_control, perturbation, expression_level_after_perturbation)
        for _ in range(num_steps):
            one_control = G.dot(one_control)
            one_control = perturb(one_control, perturbation, expression_level_after_perturbation)
        return one_control

    expression = np.column_stack(
            [all_controls[i, :] for i in range(num_controls)] + 
            [
                advance(
                    all_controls[metadata.loc[i, "matched_control"], :],
                    metadata.loc[i, "perturbation"], 
                    metadata.loc[i, "expression_level_after_perturbation"], 
                    num_steps = num_steps
                )
                for i in metadata.index if metadata.loc[i, "is_treatment"]
            ]
        ).T
    factors = np.random.random((num_features,num_features*2))
    if embed_in_higher_dimension:
        expression = factors.dot(expression)

    linear_autoregressive = anndata.AnnData(
        dtype = np.float32,
        X = expression,
        obs = metadata,
    )
    return linear_autoregressive, G, factors

hyperparameters = {
    'est_B_L1_norm':            [None],
    'true_G_L1_norm':           [None],
    'est_G_L1_norm':            [None],
    'est_G_L1_error':           [None],
    'num_epochs':               [None],
    "experiment_name":          ["S"],
    'seed':                     [0],
    'S':                        [4],
    'dimension':                [10, 20, 50, 100, 200, 500],
    'initialization_method':    ['identity'],    
    'regularization_parameter': [0.001],
    'max_epochs':               [10000],
    'batch_size':               [100000],
    "learning_rate":            ["this is overwritten below"],
    "do_early_stopping":        [True],
    "optimizer":                ["ADAM"],
    "num_controls":             [10],
    "gradient_clip_val":        [None],
    "do_shuffle":               [False],
    "do_line_search":           [True],
    "lbfgs_memory":             [100],
    "stopping_threshold":       [0.1],
    "divergence_threshold":     [np.inf],
    "profiler":                 [None],
}
conditions =  pd.DataFrame(
    [row for row in product(*hyperparameters.values())], 
    columns=hyperparameters.keys()
)
conditions["learning_rate"] = [1 if o=="L-BFGS" else 0.0005 for o in conditions["optimizer"]]

for i, _ in conditions.iterrows():
    print(f" ===== {i} ===== ")
    linear_autoregressive, G, factors = simulate_autoregressive(
        num_controls = conditions.loc[i,"num_controls"], 
        num_features = conditions.loc[i,"dimension"], 
        num_steps =    conditions.loc[i,"S"], 
        seed =         conditions.loc[i,"seed"], 
    )
    model = ggrn_backend3.GGRNAutoregressiveModel(linear_autoregressive, matching_method = "user")
    start = timer()
    trainer = model.train(
        S                        = conditions.loc[i,"S"],
        experiment_name          = conditions.loc[i,"experiment_name"],
        regression_method = "linear",
        low_dimensional_structure = "none",
        low_dimensional_training = None,
        network = None,     
        max_epochs               = conditions.loc[i, "max_epochs"],
        batch_size               = conditions.loc[i, "batch_size"],
        learning_rate            = conditions.loc[i, "learning_rate"],
        regularization_parameter = conditions.loc[i, "regularization_parameter"],
        optimizer                = conditions.loc[i, "optimizer"], 
        do_early_stopping        = conditions.loc[i, "do_early_stopping"],
        initialization_method    = conditions.loc[i, "initialization_method"],
        initialization_value = G,
        do_shuffle               = conditions.loc[i, "do_shuffle"],
        gradient_clip_val        = conditions.loc[i, "gradient_clip_val"],
        lbfgs_memory             = conditions.loc[i, "lbfgs_memory"],
        do_line_search           = conditions.loc[i, "do_line_search"],
        stopping_threshold       = conditions.loc[i, "stopping_threshold"],
        divergence_threshold     = conditions.loc[i, "divergence_threshold"],
        profiler                 = conditions.loc[i, "profiler"],
    )
    end = timer()
    conditions.loc[i, "walltime"] = end - start
    conditions.loc[i, "num_epochs"] = trainer.logger._prev_step + 1
    for n,p in model.model.named_parameters():
        if n=="G.bias":
            Bhat = p.detach().numpy()
            conditions.loc[i,"est_B_L1_norm"]  = np.abs(Bhat).mean()
        if n=="G.weight":
            Ghat = p.detach().numpy()
            conditions.loc[i,"true_G_L1_norm"] = np.abs(G).mean()
            conditions.loc[i,"est_G_L1_norm"]  = np.abs(Ghat).mean()
            conditions.loc[i,"est_G_L1_error"] = np.abs(Ghat - G).mean()
            conditions.loc[i,"true_G^S_L1_norm"] = np.abs(np.linalg.matrix_power(G, S)).mean()
            conditions.loc[i,"est_G^S_L1_norm"]  = np.abs(np.linalg.matrix_power(Ghat, S)).mean()
            conditions.loc[i,"est_G^S_L1_error"] = np.abs(np.linalg.matrix_power(Ghat, S) - np.linalg.matrix_power(G, S)).mean()

    # Tidy and save the output (once per loop to save partial progress)
    try:
        conditions.sort_values(hyperparameters["experiment_name"], inplace=True)
    except KeyError:
        pass
    conditions.to_csv("hyperparameterSweepLog.csv")
    with open("hyperparameterSweepLog.json", 'w') as jf:
        jf.write(json.dumps(hyperparameters, indent=3))



