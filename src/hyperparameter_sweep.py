


import os
try:
    PROJECT_PATH = '/home/ekernf01/Desktop/jhu/research/projects/perturbation_prediction/cell_type_knowledge_transfer/'
    os.chdir(os.path.join(PROJECT_PATH, "perturbation_benchmarking"))
except FileNotFoundError:
    PROJECT_PATH = 'gary_path'
    os.chdir(os.path.join(PROJECT_PATH, "perturbation_benchmarking"))

import sys 
import unittest
sys.path.append(os.path.expanduser(os.path.join(PROJECT_PATH, 'perturbation_benchmarking', 'src'))) 
sys.path.append(os.path.expanduser(os.path.join(PROJECT_PATH, 'network_collection', 'load_networks'))) 
import ggrn_backend3.ggrn_backend3
from itertools import product
import pandas as pd
import numpy as np
import anndata 

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
    G = np.random.random((num_features,num_features))
    # Make sure a steady state exists
    G = G / np.max(np.real(np.linalg.eigvals(G))) 

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
    'seed':[0, 1, 2],
    'S': [1,2],
    'dimension':[1],
    'B_L1_norm': [np.NaN],
    'G_L1_norm': [np.NaN],
    'G_L2_error': [np.NaN],
}
conditions =  pd.DataFrame([row for row in product(*hyperparameters.values())], 
    columns=hyperparameters.keys())
for i, _ in conditions.iterrows():
    linear_autoregressive, G, factors = simulate_autoregressive(
        num_controls=10, 
        num_features = conditions.loc[i,"dimension"], 
        num_steps=conditions.loc[i,"S"], 
        seed = 0,
    )
    model = ggrn_backend3.ggrn_backend3.GGRNAutoregressiveModel(linear_autoregressive, matching_method = "user")
    trainer = model.train(
        S = conditions.loc[i,"S"],
        regression_method = "linear",
        low_dimensional_structure = "none",
        low_dimensional_training = None,
        network = None,     
        max_epochs=800,
        learning_rate=0.0005,
        regularization_parameter = 0.001,
        optimizer = "L-BFGS",
        do_early_stopping = False,
        initialization_method = "identity",
        initialization_value = G,
        do_shuffle = False,
    )
    for n,p in model.model.named_parameters():
        if n=="G.bias":
            Bhat = p.detach().numpy()
            conditions.loc[i,"B_L1_norm"] = np.abs(Bhat).sum()
        if n=="G.weight":
            Ghat = p.detach().numpy()
            conditions.loc[i,"G_L1_norm"] = np.abs(Ghat).sum()
            conditions.loc[i,"G_L2_error"] = np.abs(Ghat - G).sum()

conditions.to_csv("hyperparameter_sweep.csv")
