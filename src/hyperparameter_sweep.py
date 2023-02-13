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
    num_controls_per_group = 2, 
    num_control_groups = 2 ,
    num_features = 5, 
    seed = 0, 
    num_steps = 1, 
    initial_state = "random",
    F_type = "random",
    include_treatment = True,
):
    if initial_state == "identity":
        num_controls_per_group = num_features
    # This is a pretty normal simulation combining controls and perturbed samples, with linear propagation of perturbation effects.
    # The weirdest thing here is we mark some samples as "treatment", even though no gene is perturbed.
    # Even tho there is no treatment, the below code will advance them forward in time. 
    # The controls are not initialized at steady state, so each control sample has expression both 
    # "before" and "after" and these may differ.
    num_treatment_types = num_features if include_treatment else 0
    num_controls = num_controls_per_group*num_control_groups
    metadata = pd.DataFrame({
        "matched_control":                      np.tile([i for i in range(num_controls)], num_treatment_types + 2),
        "is_control":                           np.repeat([True,  False], [num_controls, (num_treatment_types+1)*num_controls]),
        "is_treatment":                         np.repeat([False, True ], [num_controls, (num_treatment_types+1)*num_controls]),
        "is_steady_state":                      [False]*(num_treatment_types+2)*num_controls,
        "perturbation":                         np.repeat(["-999", "-999"] + include_treatment*[str(i) for i in range(num_treatment_types) ], num_controls),
        "expression_level_after_perturbation":  np.repeat(10, (num_treatment_types + 2)*num_controls),
    })
    metadata.index = [str(i) for i in metadata.index]
    np.random.seed(seed)
    if initial_state == "random":
        all_controls = np.random.random((num_controls, num_features))
    elif initial_state == "identity":
        all_controls = np.kron([1, 2], np.eye(num_controls_per_group)).T
    else:
        raise ValueError(f"initial_state must be 'random' or 'identity'; received {initial_state}.")

    latent_dimension = int(round(num_features/2))
    if F_type == "random":
        Q = np.eye(num_features)
        R = np.eye(num_features)
        F = G = np.random.random((num_features,num_features)) 
    elif F_type == "low_rank":
        R = np.random.random((num_features, latent_dimension))
        Q = R.T
        G = np.eye(latent_dimension)
        F = R.dot(G.dot(Q))
    else:
        raise ValueError(f"F_type must be 'random' or 'low_rank'; received {initial_state}.")
        

    def perturb(one_control, perturbation, expression_level_after_perturbation):
        x = one_control.copy()
        if int(perturbation) >= 0:
            x[int(perturbation)] = float(expression_level_after_perturbation)
        return x
    def advance(one_control, perturbation, expression_level_after_perturbation, num_steps):
        one_control = perturb(one_control, perturbation, expression_level_after_perturbation)
        for _ in range(num_steps):
            one_control = F.dot(one_control) + one_control
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
    linear_autoregressive = anndata.AnnData(
        dtype = np.float32,
        X = expression,
        obs = metadata,
    )
    return linear_autoregressive, R,G,Q,F, latent_dimension

hyperparameters = {
    # What are you doing right now? This becomes the tensorboard logging folder name.
    "experiment_name":          ["Scaling S after resnet-ification"],
    # Metrics of speed and accuracy
    'est_B_L1_norm':               [None],
    'true_F_L1_norm':              [None],
    'est_F_L1_norm':               [None],
    'est_F_L1_error':              [None],
    "est_QR_constraint_violation": [None],
    'num_epochs':                  [None],
    # Settings affecting data generation by not models
    "initial_state":            ["random"],
    "F_type":                   ["low_rank"],
    "include_treatment":        [True],
    # Settings affecting both inference and data generation
    'seed':                     [0],
    'S':                        [4],
    'dimension':                [1000],    
    "num_controls":             [10],
    # Settings affecting inference but not data generation
    "low_dimensional_structure":["RGQ"],
    "low_dimensional_training": ["supervised"],
    'G_initialization_method':  ['identity'],
    'regularization_parameter': [0.001],
    # Optimization fuckery
    'max_epochs':               [10000],
    'batch_size':               [100000],
    "learning_rate":            ["this is overwritten below"],
    "do_early_stopping":        [True],
    "optimizer":                ["ADAM"],
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
    linear_autoregressive, R,G,Q,F,latent_dimension = simulate_autoregressive(
        num_controls_per_group    = conditions.loc[i,"num_controls"], 
        num_features              = conditions.loc[i,"dimension"], 
        num_steps                 = conditions.loc[i,"S"], 
        seed                      = conditions.loc[i,"seed"], 
        initial_state             = conditions.loc[i,"initial_state"],
        F_type                    = conditions.loc[i,"F_type"],
        include_treatment         = conditions.loc[i,"include_treatment"], 
    )
    start = timer()
    model = ggrn_backend3.GGRNAutoregressiveModel(
        train_data = linear_autoregressive, 
        matching_method = "user",
        S                         = conditions.loc[i,"S"],
        regression_method = "linear",
        low_dimensional_structure = conditions.loc[i,"low_dimensional_structure"],
        low_dimensional_training  = conditions.loc[i,"low_dimensional_training"],
        low_dimensional_value     = R.T if conditions.loc[i, "low_dimensional_training"].lower() == "fixed" else latent_dimension,        
        network = None,     
    )
    trainer = model.train(
        experiment_name           = conditions.loc[i,"experiment_name"],
        max_epochs                = conditions.loc[i, "max_epochs"],
        batch_size                = conditions.loc[i, "batch_size"],
        learning_rate             = conditions.loc[i, "learning_rate"],
        regularization_parameter  = conditions.loc[i, "regularization_parameter"],
        optimizer                 = conditions.loc[i, "optimizer"], 
        do_early_stopping         = conditions.loc[i, "do_early_stopping"],
        initialization_method   = conditions.loc[i, "G_initialization_method"],
        initialization_value = G,
        do_shuffle                = conditions.loc[i, "do_shuffle"],
        gradient_clip_val         = conditions.loc[i, "gradient_clip_val"],
        lbfgs_memory              = conditions.loc[i, "lbfgs_memory"],
        do_line_search            = conditions.loc[i, "do_line_search"],
        stopping_threshold        = conditions.loc[i, "stopping_threshold"],
        divergence_threshold      = conditions.loc[i, "divergence_threshold"],
        profiler                  = conditions.loc[i, "profiler"],
    )
    end = timer()
    conditions.loc[i, "walltime"] = end - start
    conditions.loc[i, "num_epochs"] = trainer.logger._prev_step + 1
    # Extract params, with Q=R=I by default
    Rhat = Qhat = np.eye(linear_autoregressive.X.shape[1])
    for n,p in model.model.named_parameters():
        if n=="R.weight":
            Rhat = p.detach().numpy()
        if n=="Q.weight":
            Qhat = p.detach().numpy()            
        if n=="G.bias":
            Bhat = p.detach().numpy()
        if n=="G.weight":
            Ghat = p.detach().numpy()
    # Measure accuracy
    Fhat = Rhat.dot(Ghat).dot(Qhat)
    conditions.loc[i,"est_B_L1_norm"]  = np.abs(Bhat).mean()
    conditions.loc[i,"est_F_L1_error"] = np.abs(Fhat - F).mean()
    conditions.loc[i,"true_F_L1_norm"] = np.abs(F).mean()
    conditions.loc[i,"est_F_L1_norm"]  = np.abs(Fhat).mean()
    S = conditions.loc[i,"S"]
    conditions.loc[i,"true_F^S_L1_norm"] = np.abs(np.linalg.matrix_power(F, S)).mean()
    conditions.loc[i,"est_F^S_L1_norm"]  = np.abs(np.linalg.matrix_power(Fhat, S)).mean()
    conditions.loc[i,"est_F^S_L1_error"] = np.abs(np.linalg.matrix_power(Fhat, S) - np.linalg.matrix_power(F, S)).mean()
    conditions.loc[i,"est_QR_constraint_violation"] = np.abs(Qhat.dot(Rhat) - np.eye(Qhat.shape[0])).mean()
    
    # Tidy and save the output (once per loop to save partial progress)
    try:
        conditions.sort_values(hyperparameters["experiment_name"], inplace=True)
    except KeyError:
        pass
    conditions.to_csv("hyperparameterSweepLog.csv")
    with open("hyperparameterSweepLog.json", 'w') as jf:
        jf.write(json.dumps(hyperparameters, indent=3))




