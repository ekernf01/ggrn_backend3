import torch
import pytorch_lightning as pl
import ggrn_backend3._internals as linear_autoregressive
import ggrn.api as ggrn
import anndata
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import os
from datetime import datetime
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.neighbors import KDTree

class AnnDataMatchedControlsDataSet(torch.utils.data.Dataset):
    """Data loader that chops up an AnnData and feeds it to PyTorch"""
    def __init__(self, adata: anndata.AnnData, matching_method: str, assume_unrecognized_genes_are_controls: bool) -> None:
        """Data loader that chops up an AnnData and feeds it to PyTorch

        Args:
            adata (anndata.AnnData): Perturbation data.
            matching_method (str): How to match controls to each treated sample. Currently accepts "user"; future plans include "random" and "closest"
            assume_unrecognized_genes_are_controls (bool): if True, all unrecognized entries of adata.obs["perturbation"] are treated as controls. 
                If False, only "control" is allowed and any other perturbation name must be a gene name in adata.
        """
        super().__init__()
        self.assume_unrecognized_genes_are_controls = assume_unrecognized_genes_are_controls
        self.adata = adata
        self.adata.X = self.adata.X.astype(np.float32)
        self.adata.var["numeric_index"] = [i for i in range(self.adata.var.shape[0])]
        def apply_over_comma_separated_string(s: str, f, sep=","):
            return sep.join([str(f(x)) for x in s.split(sep)])
        self.adata.obs["perturbation_index"] = [
            apply_over_comma_separated_string(p, self.get_index_from_gene)
            for p in self.adata.obs["perturbation"]
        ]

        assert self.adata.X.dtype == np.float32, f"Use single precision input for this model; got type {self.adata.X.dtype}."

        # Mark each observation as treatment, control, and/or steady state
        if "is_control" not in set(self.adata.obs.columns):
            raise KeyError("train data must have a boolean column in .obs with key 'is_control'.")
        if self.adata.obs["is_control"].dtype.name != "bool":
            dt = self.adata.obs["is_control"].dtype
            raise TypeError(f"train.obs['is_control'] must be boolean. dtype: {repr(dt)}")
        if "is_treatment" not in set(self.adata.obs.columns):
            self.adata.obs["is_treatment"] = ~self.adata.obs["is_control"]
        # Default: controls are at steady state, treated samples are not
        if "is_steady_state" not in set(self.adata.obs.columns):
            self.adata.obs["is_steady_state"] = self.adata.obs["is_control"]
        else:
            assert self.adata.obs["is_steady_state"].dtype.name == "bool", "obs['is_steady_state'] must have dtype bool"
        if "perturbation" not in self.adata.obs.columns:
            raise KeyError("train data must have a comma-separated str column in .obs with key 'perturbation'.")
        if "expression_level_after_perturbation" not in self.adata.obs.columns:
            raise KeyError("train data must have a comma-separated str column in .obs with key 'expression_level_after_perturbation'.")
        if self.adata.obs["expression_level_after_perturbation"].dtype.name != "str":
            self.adata.obs['expression_level_after_perturbation'] = self.adata.obs['expression_level_after_perturbation'].astype(str)
        self.adata = MatchControls(self.adata, matching_method)
        assert self.adata.X.dtype == np.float32, f"Use single-precision input with this model; got dtype {self.adata.X.dtype}"

    def __len__(self):
        """When building batches, exclude samples with no matched 'before' timepoint.""" 
        l = self.adata.obs["matched_control"].notnull().sum()
        return l

    def __getitem__(self, idx):
        """When building batches, exclude samples with no matched 'before' timepoint."""
        idx = np.where(idx==self.adata.obs["index_among_eligible_observations"])[0][0]
        matched_idx = self.adata.obs.iat[idx,self.adata.obs.columns.get_loc("matched_control")]
        assert self.adata[idx        ,:].X.dtype == np.float32, "Use single-precision input with this model."
        assert self.adata[matched_idx,:].X.dtype == np.float32, "Use single-precision input with this model."

        return (
            self.adata.X[idx,        :],
            self.adata.obs.iloc[idx,:].to_dict(),
            self.adata[matched_idx,:].X.toarray().squeeze() 
        )
    
    def get_index_from_gene(self, g):
        if g in self.adata.var_names:
            return self.adata.var.loc[g, "numeric_index"]  
        elif g.lower()=="control" or not self.assume_unrecognized_genes_are_controls:
            return -999 #sentinel value indicates this is a control
        else:
            raise KeyError(f"Cannot find gene named {g}, and so it cannot be perturbed. Use 'control' for unperturbed observations or set assume_unrecognized_genes_are_controls=True.")

def MatchControls(train_data: anndata.AnnData, matching_method: str):
    if matching_method.lower() == "closest":
        assert "matched_control" not in train_data.obs.columns, "matched_control column is already in .obs; delete it or set matching_method='user'."
        assert "is_control" in train_data.obs.columns, "A boolean field 'is_control' is required."
        assert any(train_data.obs["is_control"]), "matching_method=='closest' requires some True entries in train_data.obs['is_control']."
        # Index the control expression with a K-D tree
        kdt = KDTree(train_data.X[train_data.obs["is_control"],:], leaf_size=30, metric='euclidean')
        control_index_converter = train_data.obs.index[train_data.obs["is_control"]].copy()
        # Query the index to get 1 nearest neighbor
        nn = [nn[0] for nn in kdt.query(train_data.X, k=1, return_distance=False)]
        train_data.obs["matched_control"] = control_index_converter[nn]
        # Mark controls as steady state, as they ought to be self-matched
        for j,i in enumerate(train_data.obs.index):
            if train_data.obs.loc[i, "is_control"]:
                train_data.obs.loc[i, "is_steady_state"] = True
                train_data.obs.loc[i, "matched_control"] = i # Controls should be self-matched, which can fail when input data have exact dupes.
    elif matching_method.lower() == "optimal_transport":
        assert "matched_control" not in train_data.obs.columns, "matched_control column is already in .obs; delete it or set matching_method='user'."
        raise NotImplementedError("Sorry, cannot yet match samples to controls by optimal transport.")
        train_data.obs["matched_control"] = "placeholder"
    elif matching_method.lower() == "random":
        assert "matched_control" not in train_data.obs.columns, "matched_control column is already in .obs; delete it or set matching_method='user'."
        train_data.obs["matched_control"] = np.random.choice(
            np.where(train_data.obs["is_control"])[0], 
            train_data.obs.shape[0], 
            replace = True,
        )
    elif matching_method.lower() == "user":
        assert "matched_control" in train_data.obs.columns, "You must provide obs['matched_control']."
        assert all( train_data.obs.loc[train_data.obs["matched_control"].notnull(), "matched_control"] < train_data.n_obs ), f"Matched control index may not be above {train_data.n_obs}."
    else: 
        raise ValueError("matching method must be 'closest' or 'random'.")
    # This helps exclude observations with no matched control.
    train_data.obs["index_among_eligible_observations"] = train_data.obs["matched_control"].notnull().cumsum()-1
    train_data.obs.loc[train_data.obs["matched_control"].isnull(), "index_among_eligible_observations"] = np.nan
    return train_data


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
        loss_function = None,
        network = None,
        assume_unrecognized_genes_are_controls = False,
    ):
        self.train_data = AnnDataMatchedControlsDataSet(train_data, matching_method, assume_unrecognized_genes_are_controls=assume_unrecognized_genes_are_controls)
        self.model = None
        self.network = network
        self.model = None,
        self.S = S
        self.loss_function = loss_function if loss_function is not None else torch.nn.L1Loss() 
        self.regression_method = regression_method
        self.low_dimensional_structure = low_dimensional_structure
        self.low_dimensional_training  = low_dimensional_training
        self.low_dimensional_value     = low_dimensional_value
        if self.low_dimensional_structure is not None and self.low_dimensional_structure.lower() != "none":
            # type and shape checking
            assert type(self.low_dimensional_value) in (int, np.ndarray, np.matrix), f"low_dimensional_value must be int, ndarray, or matrix. Received: {type(self.low_dimensional_value)}"
            if self.low_dimensional_training.lower() in ("svd"):
                assert type(self.low_dimensional_value)==int, f"If SVD is used, low_dimensional_value must be an int. Received: {type(self.low_dimensional_value)}"
            if type(self.low_dimensional_value)==int:
                assert self.low_dimensional_value < train_data.n_obs, "Latent dimension must be less than number of obs in training data"
                assert self.low_dimensional_value < train_data.n_vars, "Latent dimension must be less than number of vars in training data"
                assert self.low_dimensional_value > 0, "Latent dimension must be greater than 0"
            else:
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
        pin_memory = True,
        do_early_stopping = True,
        initialization_method = "identity",
        initialization_value: np.ndarray = None,
        do_shuffle: bool = False,
        gradient_clip_val: float = None,
        do_line_search: bool = True,
        lbfgs_memory=100,
        stopping_threshold = 0.1,
        divergence_threshold = np.inf,
        experiment_name = None,
        profiler = None,
    ):
        """_summary_

        Args:
            device (_type_, optional): "cuda" or "cpu" or None (auto-select). Defaults to None.
            max_epochs (int, optional): Maximum number of passes through the data. Defaults to 10000.
            learning_rate (float, optional): Learning rate for ADAM. Defaults to 0.001.
            batch_size (int, optional): How much data to process per batch. Defaults to 64 but try to set it higher if you can.
            regularization_parameter (float, optional): LASSO penalty param on F. Defaults to 0.
            optimizer (str, optional): "ADAM" or "L-BFGS". Defaults to "ADAM".
            num_workers (int, optional):  Passed to torch's DataLoader.
            pin_memory: Passed to torch's DataLoader.
            do_early_stopping (bool, optional): If True, stop when training loss fails to decrease 30 times in a row. 
                Defaults to True. Useful to give up early when weights diverge.
            initialization_method (str, optional): How to initialize the weights. Defaults to "identity"; other options are usually MUCH worse.
            initialization_value (np.ndarray, optional): Value to init to if initialization_method is "user". Defaults to None.
            do_shuffle (bool, optional): Shuffle the data between epochs? Defaults to False.
            gradient_clip_val (float, optional): Maximum value of each coordinate of the gradient. Defaults to None.
            do_line_search (bool, optional): Do Wolfe line search with L-BFGS? Strongly recommended. Defaults to True.
            lbfgs_memory (int, optional): Defaults to 100.
            stopping_threshold (float, optional): See pytorch_lightning.callbacks.early_stopping.EarlyStopping(). Defaults to 0.1.
            divergence_threshold (_type_, optional): See pytorch_lightning.callbacks.early_stopping.EarlyStopping(). Defaults to np.inf.
            experiment_name (str, optional): Used for logging convergence during training. Defaults to None.
            profiler (_type_, optional): Passed to pl.Trainer(). Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
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
            loss_function=self.loss_function,
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
            num_workers=num_workers, 
            pin_memory=pin_memory,
        )
        trainer = pl.Trainer(
            profiler = profiler,
            logger = None if experiment_name is None else TensorBoardLogger(
                name=experiment_name, 
                save_dir=os.path.join(
                    "lightning_logs", 
                    experiment_name,
                    datetime.now().strftime("%Y-%m-%d__%H:%M:%S"),
                )
            ),
            max_epochs=int(max_epochs),
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
        return trainer
        
    def predict(
        self,
        perturbations: list, 
        starting_expression: anndata.AnnData
    ) -> anndata.AnnData:
        """Predict future expression

        Args:
            perturbations (list): List of tuples like [("NANOG", 0), ("OCT4", 0), ("NANOG,OCT4", "0,0"), ("control", 0)]. Each tuple corresponds
                to one expression profile in the output, and comma-separated lists are used to apply multiple perturbations to the
                same predicted expression profile. To run a simulation with no perturbed genes, use "control" as the gene name.
                Tuples like ("control,2", "0,0") or ("notagene", 0) are not allowed.
            starting_expression (anndata.AnnData): Starting state for simulation. One expression profile per perturbation.

        Returns:
            anndata.AnnData: Object with one expression profile per perturbation
        """
        assert type(starting_expression) is anndata.AnnData, f"starting_expression must be AnnData; got {type(starting_expression)}"
        assert starting_expression.n_obs == len(perturbations), "Starting expression must have one obs per perturbation"
        try:
            starting_expression.X = starting_expression.X.toarray()
        except AttributeError:
            pass
        predictions = starting_expression.copy()

        # Use this when downstream code is fine with missing values among ints
        def permissive_int(x):
            try:
                return int(x)
            except ValueError:
                return np.NaN
            
        for i,p in enumerate(perturbations):
            pv = np.array([permissive_int(x)                      for x in str(p[1]).split(",")])
            pi = np.array([self.train_data.get_index_from_gene(g) for g in p[0].split(",")])
            not_control = pi != -999
            pi = pi[not_control]
            pv = pv[not_control]
            predictions[i,:] = self.predict_one_sample(
                starting_expression = starting_expression[i, :].X.toarray().squeeze(), 
                perturbed_indices = pi,
                perturbed_values  = pv,
            )
            predictions.obs.loc[
                predictions.obs.index[i], # I am afraid to do iloc with col names
                [
                    "is_control",
                    "is_treatment",
                    "perturbation",
                    "expression_level_after_perturbation",
                ]
            ] = (
                p[0].lower()=="control",
                p[0].lower()!="control",
                p[0],
                np.float(p[1]),
            )
        return predictions

    def predict_one_sample(
        self, 
        perturbed_indices: np.ndarray,
        perturbed_values: np.ndarray,
        starting_expression: np.ndarray,
    ) -> np.ndarray:
        """Predict expression after perturbation.

        Args:
            perturbed_indices (np.ndarray): Indices of perturbed features.
            perturbed_values (np.ndarray): Values of perturbed features.
            starting_expression (np.ndarray): Expression levels before perturbation.

        Returns:
            np.ndarray: Expression after perturbation. Numpy array of same shape as starting_expression.
        """
        assert type(perturbed_indices)   is np.ndarray, f"Indices of perturbed features must be provided as a numpy array; got {type(perturbed_indices)}"
        assert type(perturbed_values)    is np.ndarray, f"Values of perturbed features must be provided as a numpy array; got {type(perturbed_values)}"
        assert type(starting_expression) is np.ndarray, f"Starting values must be provided as a numpy array; got {type(starting_expression)}"
        x = starting_expression
        with torch.no_grad():
            for _ in range(self.S):
                x = self.model.forward(torch.tensor(x), zip(perturbed_indices, perturbed_values))
                for i in range(len(perturbed_indices)):
                    x[perturbed_indices[i]] = perturbed_values[i]
        return x.detach().numpy()


def simulate_autoregressive(
    num_controls_per_group = 2, 
    num_control_groups = 2 ,
    num_features = 5, 
    seed = 0, 
    num_steps = 1, 
    initial_state = "random",
    F_type = "random",
    include_treatment = True,
    latent_dimension = None,
):
    """Simulate data from an autoregressive model of the sort ggrn_backend3 assumes. 

    Args:
        num_controls_per_group (int, optional): How many control samples in each group. Defaults to 2.
        num_control_groups (int, optional): How many groups of control samples. Defaults to 2.
        num_features (int, optional): Data dimension. Defaults to 5.
        seed (int, optional): Used to make repeatable random data. Defaults to 0.
        num_steps (int, optional): Also called S in the mathematical specs. How many time-steps forward to "run" it. Defaults to 1.
        initial_state (str, optional): Can be "random" or "identity". For some purposes it's useful to start with an identity matrix. Defaults to "random".
        F_type (str, optional): F is the transition matrix -- see the mathematical specs for more details. Can be "random" or "low-rank" or "zero". 
            Defaults to "random".
        include_treatment (bool, optional): If true, copy each control sample num_features times and perturb one feature in each of the new groups.
            Defaults to True.
        latent_dimension (int, optional): rank of F if F_type is "low_rank"; otherwise not used. Defaults to half of num_features, rounded up.

    Returns:
        tuple: linear_autoregressive, R,G,Q,F, latent_dimension where linear_autoregressive is an 
            AnnData, R,G,Q,F are transition matrices or factors thereof (F=RGQ), and latent_dimension is the rank of F.
    """
    if initial_state == "identity":
        num_controls_per_group = num_features
    # This is a pretty normal simulation combining controls and perturbed samples, with linear propagation of perturbation effects.
    # The weirdest naming decision here is we mark some samples as "treatment", even though no gene is perturbed.
    # Even tho there is no treatment, the below code will only advance them forward in time if they are labeled as treated. 
    # We'll fix this right before returning the object.
    #
    # Also: we train on input-output pairs. The first element of each pair is never used as output. So, its "matched_control" must be nan.

    num_treatment_types = num_features if include_treatment else 0
    num_controls = num_controls_per_group*num_control_groups
    metadata = pd.DataFrame({
        "index": [str(i) for i in range((num_treatment_types+2)*num_controls)],
        "matched_control": np.concatenate((
            np.repeat(np.nan, num_controls), 
            np.tile([i for i in range(num_controls)], num_treatment_types + 1),
        )),
        "is_control":                           np.repeat([True,  False], [num_controls, (num_treatment_types+1)*num_controls]),
        "is_treatment":                         np.repeat([False, True ], [num_controls, (num_treatment_types+1)*num_controls]),
        "is_steady_state":                      [False]*(num_treatment_types+2)*num_controls,
        "perturbation":                         np.repeat(["-999", "-999"] + include_treatment*[str(i) for i in range(num_treatment_types) ], num_controls),
        "expression_level_after_perturbation":  np.repeat(10, (num_treatment_types + 2)*num_controls),
    })
    metadata["matched_control"] = metadata["matched_control"].round().astype('Int64')
    np.random.seed(seed)
    if initial_state == "random":
        all_controls = np.random.random((num_controls, num_features))
    elif initial_state == "identity":
        all_controls = np.kron([1, 2], np.eye(num_controls_per_group)).T
    else:
        raise ValueError(f"initial_state must be 'random' or 'identity'; received {initial_state}.")

    if latent_dimension is None:
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
    elif F_type == "zero":
        Q = np.eye(num_features)
        R = np.eye(num_features)
        F = G = np.zeros((num_features, num_features))
    else:
        raise ValueError(f"F_type must be 'random' or 'low_rank' or 'zero'; received {initial_state}.")
        

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
                for i in metadata.index if metadata["matched_control"].notnull()[i]
            ]
        ).T
    linear_autoregressive = anndata.AnnData(
        dtype = np.float32,
        X = expression,
        obs = metadata,
    )
    # Make some metadata items more suitable for external use
    linear_autoregressive.obs["perturbation"].replace("-999", "control", inplace = True)
    linear_autoregressive.obs.loc[linear_autoregressive.obs["perturbation"]=="control","is_control"] = True
    linear_autoregressive.obs.loc[linear_autoregressive.obs["perturbation"]=="control","is_treatment"] = False
    return linear_autoregressive, R,G,Q,F, latent_dimension
