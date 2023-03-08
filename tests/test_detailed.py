import ggrn_backend3.api as ggrn_backend3
from itertools import product
import pandas as pd
import numpy as np
from timeit import default_timer as timer
import unittest
def assert_equal_anndata(ad1, ad2, decimal = 7):
    np.testing.assert_almost_equal(ad1.X, ad2.X, decimal=decimal,  err_msg=".X is off")
    assert ad1.obs.equals(ad2.obs), ".obs is off"
    assert ad1.var.equals(ad2.var), ".var is off"


def set_up_trials(hyperparameters):
    conditions =  pd.DataFrame(
        [row for row in product(*hyperparameters.values())], 
        columns=hyperparameters.keys()
    )
    conditions["learning_rate"] = [1 if o=="L-BFGS" else 0.0005 for o in conditions["optimizer"]]

    # Rule out some conditions that we know are mis-specified (inference cannot fully accommodate data generation)
    expected_to_fail = (
        # If you fix F=0 and fix Q=R=I, you're gonna have a bad time, because G is absorbed into R.
        (conditions["F_type"] == "zero") & (conditions["low_dimensional_training"] == "fixed") | 
        # If you fix F=0 and X0 = I, I don't think that's identifiable, based on some tests where the inference is wrong and the predictions are fine.
        (conditions["F_type"] == "zero") & (conditions["initial_state"] == "identity") | 
        # Using a full rank F during simulation requires using a full rank F during inference
        (conditions["F_type"] == "random") & (conditions["low_dimensional_structure"] == "RGQ") | 
        # SVD is only exact here if you initialize to the identity matrix
        (conditions["initial_state"] == "random") & (conditions["low_dimensional_training"] == "SVD") | 
        # SVD is only exact here if you don't include perturbations
        (conditions["include_treatment"] == True) & (conditions["low_dimensional_training"] == "SVD") | 
        # Matching is only exact if everything is closest to the control that generated it
        # This can be guaranteed if F=0 and there are no perts.
        # Otherwise it might still work but is not easily guaranteed.
        # Another reason to set F=0: in our simulation code, the controls are NOT guaranteed to reach steady state, but
        # they will be self-matched if matching_method='closest'.
        (conditions["matching_method"] == 'closest') & ( (conditions["F_type"] != "zero") | conditions["include_treatment"] )
    )
    conditions = conditions.loc[~expected_to_fail, :]
    return conditions


# Test the remaining conditions for correct inferences
def test_correctness_in_detail(conditions, result_csv):
    for i, _ in conditions.iterrows():
        print(f" ===== Testing condition {i} ===== ", flush=True)
        print(conditions.loc[i,:].T)
        linear_autoregressive, R,G,Q,F,latent_dimension = ggrn_backend3.simulate_autoregressive(
            num_controls_per_group    = conditions.at[i,"num_controls"], 
            num_features              = conditions.at[i,"dimension"], 
            num_steps                 = conditions.at[i,"S"], 
            seed                      = conditions.at[i,"seed"], 
            initial_state             = conditions.at[i,"initial_state"],
            F_type                    = conditions.at[i,"F_type"],
            include_treatment         = conditions.at[i,"include_treatment"], 
        )
        if conditions.loc[i,"matching_method"] != "user":
            del linear_autoregressive.obs["matched_control"]
        start = timer()
        model = ggrn_backend3.GGRNAutoregressiveModel(
            train_data = linear_autoregressive, 
            matching_method =           conditions.at[i,"matching_method"],
            S                         = conditions.at[i,"S"],
            regression_method = "linear",
            low_dimensional_structure = conditions.at[i,"low_dimensional_structure"],
            low_dimensional_training  = conditions.at[i,"low_dimensional_training"],
            # Since this is a unit test, we can cheat and reveal the true R if needed.
            low_dimensional_value     = R.T if conditions.at[i, "low_dimensional_training"].lower() == "fixed" else latent_dimension,        
            network = None,     
        )

        trainer = model.train(
            experiment_name           = conditions.at[i,"experiment_name"],
            max_epochs                = conditions.at[i, "max_epochs"],
            batch_size                = conditions.at[i, "batch_size"],
            learning_rate             = conditions.at[i, "learning_rate"],
            regularization_parameter  = conditions.at[i, "regularization_parameter"],
            optimizer                 = conditions.at[i, "optimizer"], 
            do_early_stopping         = conditions.at[i, "do_early_stopping"],
            initialization_method     = conditions.at[i, "G_initialization_method"],
            initialization_value = G,
            do_shuffle                = conditions.at[i, "do_shuffle"],
            gradient_clip_val         = conditions.at[i, "gradient_clip_val"],
            lbfgs_memory              = conditions.at[i, "lbfgs_memory"],
            do_line_search            = conditions.at[i, "do_line_search"],
            stopping_threshold        = conditions.at[i, "stopping_threshold"],
            divergence_threshold      = conditions.at[i, "divergence_threshold"],
            profiler                  = conditions.at[i, "profiler"],
        )
        end = timer()

        conditions.loc[i, "num_epochs"] = trainer.current_epoch
        conditions.loc[i, "walltime"] = end - start
        conditions.loc[i, "time_per_epoch"] = conditions.loc[i,"walltime"]/conditions.loc[i,"num_epochs"]
        # Extract params, with G=Q=R=I by default
        Rhat = Qhat = np.eye(linear_autoregressive.X.shape[1])
        Ghat = np.eye(latent_dimension)
        Bhat = np.zeros(latent_dimension)
        for n,p in model.model.named_parameters():
            if n=="R.weight":
                Rhat = p.detach().numpy()
            if n=="Q.weight":
                Qhat = p.detach().numpy()            
            if n=="G.bias":
                Bhat = p.detach().numpy()
            if n=="G.weight":
                Ghat = p.detach().numpy()
        # Measure inference accuracy
        Fhat = Rhat.dot(Ghat).dot(Qhat)
        conditions.loc[i,"est_B_L1_norm"]  = np.abs(Bhat).mean()
        conditions.loc[i,"est_F_L1_error"] = np.abs(Fhat - F).mean()
        conditions.loc[i,"true_F_L1_norm"] = np.abs(F).mean()
        conditions.loc[i,"est_F_L1_norm"]  = np.abs(Fhat).mean()
        S = conditions.loc[i,"S"]
        conditions.loc[i,"true_F^S_L1_norm"] = np.abs(np.linalg.matrix_power(   F, int(S))).mean()
        conditions.loc[i,"est_F^S_L1_norm" ] = np.abs(np.linalg.matrix_power(Fhat, int(S))).mean()
        conditions.loc[i,"est_F^S_L1_error"] = np.abs(np.linalg.matrix_power(Fhat, int(S)) - np.linalg.matrix_power(F, S)).mean()

        # Tidy and save the output (once per loop to save partial progress)
        print(conditions.loc[i,:].T)
        conditions.to_csv(result_csv)

        # This is the easiest thing to mathematically guarantee. The rest may not be identifiable.
        print(f"Error in coefs: {conditions.loc[i,'est_F^S_L1_error']}")
        np.testing.assert_almost_equal(conditions.loc[i,"est_F^S_L1_error"], 0*conditions.loc[i,"est_F^S_L1_error"], decimal = 1)

        # Check predictions (after fixing/removing some obs fields that are not guaranteed to match)
        end_state = linear_autoregressive[-1,:]
        perturbations = [(end_state.obs["perturbation"][0], end_state.obs["expression_level_after_perturbation"][0])]
        predictions = model.predict(
            perturbations = perturbations,
            starting_expression = linear_autoregressive[int(end_state.obs["matched_control"]),:],
        )
        predictions.obs.index = end_state.obs.index
        del end_state.obs  ["perturbation_index"]
        del predictions.obs["perturbation_index"]
        del end_state.obs  ["index_among_eligible_observations"]
        del predictions.obs["index_among_eligible_observations"]
        del end_state.obs  ["index"]
        del predictions.obs["index"]
        del end_state.obs  ["matched_control"]
        del predictions.obs["matched_control"]
        print(end_state.obs)
        print(predictions.obs)
        assert_equal_anndata(predictions, end_state.copy(), decimal=0)       



# Choose combos of settings to test and put them in a big dataframe.
correctness_test_config = set_up_trials(
    hyperparameters = {
        # What are you doing right now? This becomes the tensorboard logging folder name.
        "experiment_name":          ["unit test"],
        # Metrics of speed and accuracy
        'est_B_L1_norm':               [None],
        'true_F_L1_norm':              [None],
        'est_F_L1_norm':               [None],
        'est_F_L1_error':              [None],
        'num_epochs':                  [None],
        'time_per_epoch':              [None],
        # Settings affecting data generation but not inference
        "F_type":                   ["low_rank", "random", "zero"],
        "initial_state":            ["random", "identity"],  
        "include_treatment":        [True, False],           
        # Settings affecting both inference and data generation
        'seed':                     [0],
        'S':                        [1],
        'dimension':                [4],    
        "num_controls":             [10],
        # Settings affecting inference but not data generation
        "matching_method":          ["closest", "user"],
        "low_dimensional_structure":["RGQ", "none"],
        "low_dimensional_training": ["supervised", "SVD", "fixed"],
        'G_initialization_method':  ['identity'],
        'regularization_parameter': [0.0001],
        # Optimization fuckery
        'max_epochs':               [10000],
        'batch_size':               [100000],
        "learning_rate":            ["this is overwritten below"],
        "do_early_stopping":        [True],
        "optimizer":                ["L-BFGS"],
        "gradient_clip_val":        [None],
        "do_shuffle":               [False],
        "do_line_search":           [True],
        "lbfgs_memory":             [100],
        "stopping_threshold":       [0.001],
        "divergence_threshold":     [np.inf],
        "profiler":                 [None],
    }
)

# Choose combos of settings to test and put them in a big dataframe.
speed_test_config = set_up_trials(
    hyperparameters = {
        # What are you doing right now? This becomes the tensorboard logging folder name.
        "experiment_name":          ["unit test"],
        # Metrics of speed and accuracy
        'est_B_L1_norm':               [None],
        'true_F_L1_norm':              [None],
        'est_F_L1_norm':               [None],
        'est_F_L1_error':              [None],
        'num_epochs':                  [None],
        # Settings affecting data generation but not inference
        "F_type":                   ["low_rank"],
        "initial_state":            ["random"],  
        "include_treatment":        [True],           
        # Settings affecting both inference and data generation
        'seed':                     [0],
        'S':                        [1],
        'dimension':                [1000],    
        "num_controls":             [2],
        # Settings affecting inference but not data generation
        "matching_method":          ["user"],
        "low_dimensional_structure":["RGQ"],
        "low_dimensional_training": ["supervised"],
        'G_initialization_method':  ['identity'],
        'regularization_parameter': [0.000000001],
        # Optimization fuckery
        'max_epochs':               [10],
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
        "profiler":                 ["simple"],
    }
)
class TestBackend3(unittest.TestCase):
    # def test_correctness(self):
    #     test_correctness_in_detail(conditions=correctness_test_config, result_csv="ggrnBackend3UnitTestDetails.csv")
    def test_speed(self):
        test_correctness_in_detail(speed_test_config, result_csv="ggrnBackend3SpeedTest.csv")

if __name__ == '__main__':
    unittest.main()
