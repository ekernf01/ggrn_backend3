import os
import unittest
import ggrn_backend3.api as ggrn_backend3
import load_networks
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import anndata

example_data = sc.read_h5ad("../accessory_data/nakatake.h5ad")
example_network = load_networks.LightNetwork(files=["../accessory_data/human_promoters.parquet"])
example_perturbations = (("1", 0), ("2", 0))

# TO DO (these are implemented but not unit-tested):
# 
# - throw errors when asked to perturb a gene you have not observed
# - throw errors when latent dim is too big or small
# 

class TestBackend3(unittest.TestCase):

    def test_everything_runs(self):
        """Make sure all user-facing options run without errors"""
        linear_autoregressive, R,G,Q,F, factors = ggrn_backend3.simulate_autoregressive(num_controls_per_group=10, num_features = 3)
        for matching_method in [
            "user",
            # "random", 
            # "closest",
        ]:
            for regression_method in ["linear"]: #, "multilayer_perceptron"]:
                for low_dimensional_structure in ["none", "RGQ"]:
                    if low_dimensional_structure == "RGQ":
                        low_dimensional_trainings = ["supervised", "SVD", "fixed"]
                    else:
                        low_dimensional_trainings = [None]
                    for low_dimensional_training in low_dimensional_trainings:
                        model = ggrn_backend3.GGRNAutoregressiveModel(
                            linear_autoregressive, 
                            matching_method = matching_method,
                            S = 1, 
                            regression_method = regression_method,
                            low_dimensional_structure = low_dimensional_structure,
                            low_dimensional_training = low_dimensional_training,    
                            low_dimensional_value = 1,
                            network = example_network if low_dimensional_training=="fixed" else None,
                        )
                        model.train(
                            optimizer="L-BFGS",
                            max_epochs = 2,
                        )
                        y = model.predict(example_perturbations, starting_expression = linear_autoregressive[0:2,:])
                        assert type(y) == anndata.AnnData

    def test_matching_and_loading(self):
        """Make sure we can pair up everything with a control"""
        td = ggrn_backend3.MatchControls(example_data, "random")
        # td = ggrn_backend3.MatchControls(train_data, "closest")
        torchy_dataset = ggrn_backend3.AnnDataMatchedControlsDataSet(td, "user")
        torchy_dataloader = torch.utils.data.DataLoader(torchy_dataset)
        x = torchy_dataset[0]
        batch = next(iter(torchy_dataloader))
        assert "treatment" in x.keys()
        assert "matched_control" in x.keys()
        assert "expression" in x["treatment"].keys()
        assert "metadata"   in x["treatment"].keys()
        assert "expression" in x["matched_control"].keys()
        assert "metadata"   in x["matched_control"].keys()
        assert all(
            [f in x["treatment"]["metadata"].keys() 
            for f in {"is_control", "is_treatment", "is_steady_state", "perturbation_index", "expression_level_after_perturbation"}]
        )

if __name__ == '__main__':
    unittest.main()