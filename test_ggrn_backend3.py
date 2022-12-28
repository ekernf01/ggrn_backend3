


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
import load_networks
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import anndata

example_data = sc.read_h5ad("../accessory_data/nakatake.h5ad")
example_network = load_networks.LightNetwork(files=["../accessory_data/human_promoters.parquet"])
example_perturbations = (("KLF8", 0), ("GATA1", 0), ("empty_vector", np.nan))


class TestBackend3(unittest.TestCase):
    def test_matching_and_loading(self):
        td = ggrn_backend3.ggrn_backend3.MatchControls(example_data, "random")
        # td = ggrn_backend3.ggrn_backend3.MatchControls(train_data, "closest")
        torchy_dataset = ggrn_backend3.ggrn_backend3.AnnDataMatchedControlsDataSet(td, "user")
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


    def test_user_interface(self):
        linear_autoregressive, G, factors = simulate_autoregressive(num_controls=10, num_features = 3)
        for matching_method in [
            "user",
            # "random", 
            # "closest",
        ]:
            for regression_method in ["linear"]: #, "multilayer_perceptron"]:
                for low_dimensional_structure in ["none"]: #, "RGQ"]:
                    if low_dimensional_structure == "RGQ":
                        low_dimensional_trainings = ["supervised", "PCA", "fixed"]
                    else:
                        low_dimensional_trainings = [None]
                    for low_dimensional_training in low_dimensional_trainings:
                        model = ggrn_backend3.ggrn_backend3.GGRNAutoregressiveModel(linear_autoregressive, matching_method = matching_method)
                        model.train(
                            S = 1,
                            regression_method = regression_method,
                            low_dimensional_structure = low_dimensional_structure,
                            low_dimensional_training = low_dimensional_training,
                            network = example_network if low_dimensional_training=="fixed" else None,
                            optimizer="L-BFGS",
                        )
                        y = model.predict(example_perturbations)

if __name__ == '__main__':
    unittest.main()
