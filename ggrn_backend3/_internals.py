import torch
import torch.nn.init as init
import pytorch_lightning as pl
import numpy as np
import math 

def P(x: torch.tensor, perturbations: list) -> torch.tensor:
    """Enact a perturbation.
    Args:
        x (torch.tensor): Gene expression vector
        p (list): list of tuples (target gene index, level), e.g. (5, 0) where gene 5 is Nanog and you knocked out Nanog.

    Returns:
        torch.tensor: Gene expression with targeted gene set to prescribed level. 
    """
    for p in perturbations:
        x[p[0]] = float(p[1])
    return x


class LinearAutoregressive(pl.LightningModule):
    def __init__(
        self, 
        n_genes,
        S,
        regression_method,
        low_dimensional_structure: str,
        low_dimensional_training: str,
        low_dimensional_value,
        loss_function: str,
        learning_rate: float, 
        regularization_parameter: float,
        optimizer: str,
        initialization_method: str,
        initialization_value: np.matrix,
        do_line_search: bool,
        lbfgs_memory: int,
    ):
        """Linear autoregressive model

        Args:
            n_genes (_type_): Data dimension
            S (int): Number of time-steps separating each sample from its matched control.
            regression_method (str): Currently only allows "linear".
            low_dimensional_structure (str) "none" or "dynamics" or "RGQ". 
                - If "none", dynamics will be modeled using the original data.
                - If "dynamics", dynamics will be modeled in a linear subspace. See also low_dimensional_training.
                - "RGQ" is a deprecated option identical to "dynamics".
            low_dimensional_training (str): "SVD" or "fixed" or "supervised". How to learn the linear subspace. 
                If "SVD", perform an SVD on the data.
                If "fixed", use a user-provided projection matrix.
                If "supervised", learn the projection and its (approximate) inverse via backprop.
            low_dimensional_value (int or numpy matrix): Dimension of the linear subspace, or an entire projection matrix.
            loss_function: Measures deviation between observed and predicted, e.g. torch.nn.HuberLoss().
            learning_rate (float): Learning rate for ADAM optimizer
            regularization_parameter (float): LASSO penalty on coefficients
            optimizer (str): 'ADAM' or 'L-BFGS' or 'ADAMW' or 'AMSGRAD'
            initialization_method (str): one of "kaiming", "he", "identity", "user". How to initialize the coefficients in the transition matrix G.
            initialization_value (np.matrix): Initial value of G, used if initialization_method=="user".  
            do_line_search (bool): Whether to do the line search in L-BFGS. We strongly recommend "True".
            lbfgs_memory (int): Memory parameter for L-BFGS.

        """
        super().__init__()
        self.S = S
        self.learning_rate = learning_rate
        self.regularization_parameter = regularization_parameter
        self.optimizer = optimizer
        self.do_line_search = do_line_search
        self.lbfgs_memory = lbfgs_memory
        self.low_dimensional_training = low_dimensional_training
        self.loss_function = loss_function

        # Set up layer sizes, e.g. n_genes -> latent_dimension -> n_genes
        if low_dimensional_structure is None or low_dimensional_structure.lower() == "none":
            self.Q = torch.nn.Identity(dtype = torch.float32)
            self.R = torch.nn.Identity(dtype = torch.float32)
            if low_dimensional_training is not None:
                print("low_dimensional_training will be ignored since low_dimensional_structure is 'none'.")
                low_dimensional_training = None
            if low_dimensional_value is not None:
                print("low_dimensional_value will be ignored since low_dimensional_structure is 'none'.")
                low_dimensional_value = None
        elif low_dimensional_structure.lower() in ["rgq", "dynamics"]:
            if low_dimensional_value is None:
                raise ValueError("low_dimensional_value must be the latent dimension, or a matrix of shape (latent_dim by n_genes).")
            try:
                latent_dimension = low_dimensional_value.shape[0]
                assert latent_dimension <= n_genes, "latent dimension must be less than or equal to n_genes"
            except AttributeError:
                latent_dimension = low_dimensional_value
                assert latent_dimension <= n_genes, "latent dimension must be less than or equal to n_genes"
            # Note there is no bias in these layers.
            # Regarding the shape:
            # A Linear layer takes input dimension before output, so if
            # L(x) = Ax and A is an N by G matrix, then L takes G first, not N.
            self.Q = torch.nn.Linear(n_genes, latent_dimension, bias=False, dtype = torch.float32)
            self.R = torch.nn.Linear(latent_dimension, n_genes, bias=False, dtype = torch.float32)
        else:
            raise ValueError(f"low_dimensional_structure must be 'none' or 'dynamics'; received {low_dimensional_structure}")

        # Set up G, which projects forward one time-step in the latent space
        if regression_method.lower() == "linear":
            if low_dimensional_structure is not None and low_dimensional_structure.lower() != "none":
                self.G = torch.nn.Linear(self.Q.weight.shape[0], self.Q.weight.shape[0], dtype=torch.float32)
            else:
                self.G = torch.nn.Linear(n_genes, n_genes, dtype=torch.float32)

        elif regression_method.lower() in {'multilayer_perceptron', 'mlp'}:
            raise NotImplementedError("mlp is not implemented yet.")
        else:
            raise ValueError(f"regression_method must be 'linear' or 'mlp'. Value: {regression_method}")
        self.initialize_g(initialization_method = initialization_method, initialization_value = initialization_value)

        # Decide how to train & initialize the projection operator Q and its right inverse approximator R
        if str(low_dimensional_structure).lower() in ["rgq", "dynamics"]: 
            if low_dimensional_training is None:
                pass 
            elif low_dimensional_training.lower() == "supervised":
                # Absorb G into R
                self.G = torch.nn.Identity()
                # pytorch will optimize Q,R 
                # we just need to initialize the weights.
                # We start with random or user-provided R and we set Q = pinv(R). 
                assert low_dimensional_value is not None
                try:
                    low_dimensional_value = np.random.normal(size=self.Q.weight.shape) * math.sqrt(2) / math.sqrt(int(low_dimensional_value))
                except TypeError: #it might already be a whole-ass matrix
                    pass
                # If we don't copy here, we get a weird error: "view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead."
                low_dimensional_value = low_dimensional_value.T.copy()
                self.R.weight = torch.nn.Parameter( torch.as_tensor(                 low_dimensional_value.astype(np.float32)   ) ) 
                self.Q.weight = torch.nn.Parameter( torch.as_tensor( np.linalg.pinv( low_dimensional_value ).astype(np.float32) ) )
            elif low_dimensional_training.lower() in ("fixed", "svd"):
                # Make sure caller has provided projection matrix
                # Fix R = pinv(Q) and tell torch not to update Q,R
                assert low_dimensional_value is not None, "Please provide a projection matrix (e.g. motif counts)."
                assert type(low_dimensional_value) != int, "Please provide a projection matrix (e.g. motif counts), not just a dimension."
                np.testing.assert_array_equal(
                    low_dimensional_value.T.shape,      
                    self.R.weight.shape, 
                    err_msg = "low_dimensional_value must be an array with dimension (latent_dimension by n_genes)."
                )
                low_dimensional_value = low_dimensional_value.astype(np.float32)
                self.R.weight = torch.nn.Parameter( torch.as_tensor(                 low_dimensional_value.T   ) ) 
                self.Q.weight = torch.nn.Parameter( torch.tensor( np.linalg.pinv( low_dimensional_value.T ) ) )
                self.Q.requires_grad_(False)
                self.R.requires_grad_(False)
            else:
                raise ValueError(f"low_dimensional_training must be 'supervised' or 'fixed' or 'SVD'. Value: {low_dimensional_training}")
        
        

    def forward(self, x, perturbations):
        x = P(x, perturbations)
        r = self.Q(x)
        r = self.G(r)
        r = self.R(r)
        x = x + r
        x = P(x, perturbations)
        return x

    def training_step(self, input_batch):
        rmse = 0
        expression, metadata, matched_control_expression = input_batch
        batch_size = len(metadata['perturbation'])
        for i in range(batch_size): 
            perturbed_indices = [int(g) for g in metadata["perturbation_index"][i].split(",")]
            perturbations = zip(
                perturbed_indices,
                [float(x) for x in metadata["expression_level_after_perturbation"][i].split(",")],
            )
            perturbations = [p for p in perturbations if p[0]!=-999] # sentinel value -999 indicates this is a control sample
            mask  = torch.as_tensor( [
                0 if g in perturbed_indices else 1 
                for g in range(len(expression[i]))
            ])
            if metadata["is_steady_state"][i]: 
                # (electric slide voice) one hop this time. *BEWMP*
                x_t = expression[i].clone()
                x_t = self(x_t, perturbations)
                rmse += self.loss_function(mask*expression[i], x_t)
            else:
                # (electric slide voice) S hops this time. *BEWMP* *BEWMP* *BEWMP* *BEWMP* 
                x_t = matched_control_expression[i].clone()
                for _ in range(self.S):
                    x_t = self(x_t, perturbations)
                rmse += self.loss_function(mask*expression[i], x_t)

        lasso_term = 0.0
        for weights_to_penalize in (
            self.G.named_parameters(),
            self.Q.named_parameters(),
            self.R.named_parameters(),
        ):
            for name, param in weights_to_penalize:
                if name=="weight": #skip bias and skip Identity blocks
                    lasso_term += torch.abs(param).sum()
        loss_value = rmse + self.regularization_parameter*lasso_term
        self.log("rmse", rmse, batch_size = batch_size, logger=True)
        self.log("lasso_term", lasso_term, batch_size = batch_size, logger=True)
        self.log("training_loss", loss_value, batch_size = batch_size, logger=True)
        return loss_value

    def configure_optimizers(self):
        if self.optimizer.upper() == "L-BFGS":
            return torch.optim.LBFGS(self.parameters(), lr=self.learning_rate, line_search_fn = 'strong_wolfe' if self.do_line_search else None, history_size=self.lbfgs_memory )        
        elif self.optimizer.upper() == "ADAM":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer.upper() == "ADAMW":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01, amsgrad=False)
        elif self.optimizer.upper() == "AMSGRAD":
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01, amsgrad=True)
        else:
            raise ValueError(f"Optimizer must be 'ADAM' or 'L-BFGS'. Value: {self.optimizer}")

    def initialize_g(self, initialization_method, initialization_value = None):
        for name, param in self.named_parameters():
            if not name.startswith("G"):
                continue
            if initialization_method.lower() in {"kaiming", "he"}:
                # Initialization suitable for use with leaky ReLU, from Kaiming He et al 2015.
                # From the official Pytorch website, https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/03-initialization-and-optimization.html
                # Original code author: Phillip Lippe
                # Original code license: CC-by-SA
                if name.endswith(".bias"):
                    init.zeros_(param)
                else:
                    init.kaiming_uniform_(param)
            elif initialization_method.lower() in {"identity"}:
                if name.endswith(".bias"):
                    init.zeros_(param)
                else:
                    init.eye_(param) 
            elif initialization_method.lower() in {"user"}:
                if not type(initialization_value) == np.ndarray:
                    raise TypeError(f"Expected np.ndarray for initialization_value; received {type(initialization_value)}")
                if not initialization_value.shape == (param.shape[0],param.shape[0]):
                    raise np.linalg.LinAlgError(f"Shape for initialization_value must be {(param.shape[0],param.shape[0])}. Received {initialization_value.shape}.")
                if name.endswith(".bias"):
                    init.zeros_(param)
                else:
                    init.constant_(param, torch.tensor(initialization_value))            
            else:
                raise ValueError(f"'initialization_method' may be 'kaiming' or 'he' or 'identity' or 'user'; received {initialization_method}")
