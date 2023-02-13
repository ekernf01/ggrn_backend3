import torch
import pytorch_lightning as pl
import linear_autoregressive
import anndata
import math
import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD
import os
from datetime import datetime
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

class AnnDataMatchedControlsDataSet(torch.utils.data.Dataset):
    def __init__(self, adata: anndata.AnnData, matching_method: str) -> None:
        super().__init__()
        self.adata = adata
        self.adata.X.astype(np.float32)
        self.adata.var["numeric_index"] = [i for i in range(self.adata.var.shape[0])]
        def apply_over_comma_separated_string(s: str, f, sep=","):
            return sep.join([str(f(x)) for x in s.split(sep)])
        self.adata.obs["perturbation_index"] = [
            apply_over_comma_separated_string(p, self.get_index_from_gene)
            for p in self.adata.obs["perturbation"]
        ]

        # Mark each observation as treatment, control, and/or steady state
        if "is_control" not in set(self.adata.obs.columns):
            raise KeyError("train data must have a boolean column in .obs with key 'is_control'.")
        if self.adata.obs["is_control"].dtype.name != "bool":
            dt = self.adata.obs["is_control"].dtype
            raise TypeError(f"train.obs['is_control'] must be boolean. dtype: {repr(dt)}")
        if "is_treatment" not in set(self.adata.obs.columns):
            self.adata.obs["is_treatment"] = ~self.adata.obs["is_control"]
        if "is_steady_state" not in set(self.adata.obs.columns):
            self.adata.obs["is_steady_state"] = self.adata.obs["is_control"]
        if "perturbation" not in self.adata.obs.columns:
            raise KeyError("train data must have a comma-separated str column in .obs with key 'perturbation'.")
        if "expression_level_after_perturbation" not in self.adata.obs.columns:
            raise KeyError("train data must have a comma-separated str column in .obs with key 'expression_level_after_perturbation'.")
        if self.adata.obs["expression_level_after_perturbation"].dtype.name != "str":
            self.adata.obs['expression_level_after_perturbation'] = self.adata.obs['expression_level_after_perturbation'].astype(str)
        self.adata = MatchControls(self.adata, matching_method)

    def __len__(self):
        return self.adata.obs.shape[0]

    def __getitem__(self, idx):
        matched_idx = self.adata.obs["matched_control"][idx]
        return {
            "treatment":      {
                "expression": self.adata.X[idx,        :], 
                "metadata": self.adata.obs.iloc[idx,:].to_dict(),
            },
            "matched_control":{
                "expression":        self.adata.X[matched_idx,:],
                "metadata":   self.adata.obs.iloc[matched_idx,:].to_dict(),
            },
        }
    
    def get_index_from_gene(self, g):
        return self.adata.var.loc[g, "numeric_index"] if g in self.adata.var_names else -999 #sentinel value indicates this is a control

def MatchControls(train_data: anndata.AnnData, matching_method: str):
    if matching_method.lower() == "closest":
        assert "matched_control" not in train_data.obs.columns, "Matched controls already present; set matching_method='user'."
        raise NotImplementedError("Cannot yet match to closest control.")
        train_data.obs["matched_control"] = "placeholder"
    elif matching_method.lower() == "random":
        assert "matched_control" not in train_data.obs.columns, "Matched controls already present; set matching_method='user'."
        train_data.obs["matched_control"] = np.random.choice(
            np.where(train_data.obs["is_control"])[0], 
            train_data.obs.shape[0], 
            replace = True,
        )
    elif matching_method.lower() == "user":
        assert "matched_control" in train_data.obs.columns
    else: 
        raise ValueError("matching method must be 'closest' or 'random'.")
    return train_data

class GradNormCallback(pl.Callback):
    """
    Logs the gradient norm.
    """

    def on_after_backward(self, trainer, model):
        model.log("my_model/grad_norm", self.gradient_norm(model))

    def gradient_norm(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

class GGRNAutoregressiveModel:

    def __init__(
        self, 
        train_data, 
        matching_method,
        S: int = 1,
        regression_method: str = "linear",
        low_dimensional_structure: str = "none",
        low_dimensional_training: str = "SVD",
        low_dimensional_value = 20,
        network = None,
    ):
        self.train_data = AnnDataMatchedControlsDataSet(train_data, matching_method)
        self.model = None
        self.network = network
        self.model = None,
        self.S = S
        self.regression_method = regression_method
        self.low_dimensional_structure = low_dimensional_structure
        self.low_dimensional_training  = low_dimensional_training
        self.low_dimensional_value     = low_dimensional_value
        if self.low_dimensional_structure is not None and self.low_dimensional_structure.lower() != "none":
            # type and shape checking
            assert type(self.low_dimensional_value) in (int, np.ndarray, np.matrix), f"low_dimensional_value must be int, ndarray, or matrix. Received: {type(self.low_dimensional_value)}"
            if self.low_dimensional_training.lower() in ("svd"):
                assert type(self.low_dimensional_value)==int, f"If SVD is used, low_dimensional_value must be an int. Received: {type(self.low_dimensional_value)}"
            if type(self.low_dimensional_value)!=int:
                assert self.low_dimensional_value.shape[1] == self.train_data.adata.X.shape[1], f"Projection matrix must be latent_dim by n_genes; received shape: {low_dimensional_value.shape}"
            # Initialize to SVD if user provides just the dimension
            if type(self.low_dimensional_value)==int:
                svd = TruncatedSVD(n_components=self.low_dimensional_value, random_state=0)
                svd.fit(train_data.X)
                self.low_dimensional_value = svd.components_       
        return

    def train(
        self,
        device = None,
        max_epochs: int = 10000, 
        learning_rate:float = 0.001,
        batch_size:int = 64,
        regularization_parameter:float = 0,
        optimizer:str = "ADAM",
        num_workers = 1,
        do_early_stopping = True,
        initialization_method = "identity",
        initialization_value = None,
        do_shuffle = False,
        gradient_clip_val = None,
        do_line_search = True,
        lbfgs_memory=100,
        stopping_threshold = 0.1,
        divergence_threshold = np.inf,
        experiment_name = None,
        profiler = None,
    ):
        if optimizer.lower() == "l-bfgs" and gradient_clip_val is not None:
            raise ValueError("Gradient clipping is not allowed with a second-order method like L-BFGS.")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = linear_autoregressive.LinearAutoregressive(
            n_genes = self.train_data.adata.X.shape[1],
            S = self.S,
            regression_method = self.regression_method,
            low_dimensional_structure = self.low_dimensional_structure,
            low_dimensional_training = self.low_dimensional_training,
            low_dimensional_value = self.low_dimensional_value,
            learning_rate = learning_rate, 
            regularization_parameter = regularization_parameter,
            optimizer=optimizer,
            initialization_method=initialization_method,
            initialization_value=initialization_value,
            do_line_search = do_line_search,
            lbfgs_memory = lbfgs_memory,
        )
        dataloader = torch.utils.data.DataLoader(
            self.train_data, 
            batch_size=int(batch_size), 
            shuffle=do_shuffle,
            num_workers=num_workers
        )
        trainer = pl.Trainer(
            profiler = profiler,
            logger = TensorBoardLogger(
                name=experiment_name, 
                save_dir=os.path.join(
                    "lightning_logs", 
                    experiment_name,
                    datetime.now().strftime("%Y-%m-%d__%H:%M:%S"),
                )
            ),
            max_epochs=max_epochs,
            accelerator=device,   
            track_grad_norm = 2, 
            gradient_clip_val = gradient_clip_val,
            deterministic = True,
            callbacks= (
                [
                    EarlyStopping(
                        monitor="training_loss", 
                        mode="min",
                        min_delta=0.00, 
                        patience=30, 
                        verbose=False,
                        strict=True,
                        stopping_threshold = stopping_threshold,
                        divergence_threshold = divergence_threshold,
                        check_finite = True,
                    )
                ] if do_early_stopping else []
                ),
        )
        trainer.fit(model=self.model, train_dataloaders=dataloader)
        trainer.save_checkpoint('./checkpoints/last.ckpt')
        # To do: resume training
        if False:
            pl.Trainer(max_epochs=1600, resume_from_checkpoint='./checkpoints/last.ckpt')

        return trainer
        
    def predict(
        self, 
        example_perturbations, 
        ):
        # TODO: select controls
        # TODO: run model forwards
        # self.model()
        return

