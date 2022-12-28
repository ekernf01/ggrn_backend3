import torch
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
        x[p[0]] = p[1]
    return x


class LinearAutoregressive(pl.LightningModule):
    def __init__(
        self, 
        n_genes,
        S,
        regression_method,
        low_dimensional_structure,
        low_dimensional_training,
        learning_rate, 
        regularization_parameter,
        optimizer,
        initialization_method,
        initialization_value,
    ):
        super().__init__()
        self.S = S
        self.learning_rate = learning_rate
        self.regularization_parameter = regularization_parameter
        self.optimizer = optimizer
        if regression_method == "linear":
            self.G = torch.nn.Linear(n_genes, n_genes)
        elif regression_method in {'multilayer_perceptron', 'mlp'}:
            raise NotImplementedError("mlp is not implemented yet.")
        else:
            raise ValueError(f"regression_method must be 'linear' or 'mlp'. Value: {regression_method}")
        
        if low_dimensional_structure in "none":
            if low_dimensional_training is not None:
                print("low_dimensional_training will be ignored since low_dimensional_structure is 'none'.")
        elif low_dimensional_structure in "RGQ":
            raise NotImplementedError("Low-D structure is not implemented yet.")
        else:
            raise ValueError("regression_method must be 'none' or 'RGQ'")

        if low_dimensional_training is None:
            pass
        elif low_dimensional_training == "supervised":
            raise NotImplementedError()
        elif low_dimensional_training == "PCA":
            raise NotImplementedError()
        elif low_dimensional_training == "fixed":
            raise NotImplementedError()
        else:
            raise ValueError(f"low_dimensional_training must be 'supervised','PCA', 'fixed'. Value: {low_dimensional_training}")
        
        self.initialize_g(initialization_method = initialization_method, initialization_value = initialization_value)
        

    def forward(self, x, perturbations):
        return P(self.G(P(x, perturbations)), perturbations)

    def training_step(self, input_batch):
        loss = 0
        for i in range(len(input_batch["treatment"]["metadata"]["perturbation"])):
            perturbed_indices = [int(g)   for g in input_batch["treatment"]["metadata"]["perturbation_index"][i].split(",")]
            perturbations = zip(
                perturbed_indices,
                [float(x) for x in input_batch["treatment"]["metadata"]["expression_level_after_perturbation"][i].split(",")],
            )
            perturbations = [p for p in perturbations if p[0]!=-999] #sentinel value indicates this is a control sample
            mask  = torch.tensor( [
                0 if g in perturbed_indices else 1 
                for g in range(len(input_batch["treatment"]["expression"][i]))
            ])
            if input_batch["treatment"]["metadata"]["is_steady_state"][i]: 
                # (electric slide voice) one hop this time. *BEWMP*
                x_t = input_batch["treatment"]["expression"][i].clone()
                x_t = self(x_t, perturbations)
                loss += torch.linalg.norm(mask*(input_batch["treatment"]["expression"][i] - x_t))
            if input_batch["treatment"]["metadata"]["is_treatment"][i]: 
                # (electric slide voice) S hops this time. *BEWMP* *BEWMP* *BEWMP* *BEWMP* 
                x_t = input_batch["matched_control"]["expression"][i].clone()
                for _ in range(self.S):
                    x_t = self(x_t, perturbations)
                loss += torch.linalg.norm(mask*(input_batch["treatment"]["expression"][i] - x_t))
        lasso_term = torch.abs(
            [param - torch.eye(param.shape[0]) if name == "weight" else param for name, param in self.G.named_parameters() ][0]
        ).sum()
        self.log("mse", loss, logger=False)
        loss += self.regularization_parameter*lasso_term
        self.log("training_loss", loss, logger=False)
        self.log("lasso_term", lasso_term, logger=False)
        return loss

    def configure_optimizers(self):
        if self.optimizer == "L-BFGS":
            return torch.optim.LBFGS(self.parameters(), lr=self.learning_rate)        
        elif self.optimizer == "ADAM":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Optimizer must be 'ADAM' or 'L-BFGS'. Value: {self.optimizer}")

    def initialize_g(self, initialization_method, initialization_value = None):
        print(" ===== Initializing G ===== ")
        for name, param in self.named_parameters():
            print(name)
            if initialization_method in {"kaiming", "he"}:
                # Initialization suitable for use with leaky ReLU, from Kaiming He et al 2015.
                # From the official Pytorch website, https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/03-initialization-and-optimization.html
                # Original code author: Phillip Lippe
                # Original code license: CC-by-SA
                if name.endswith(".bias"):
                    param.data.fill_(0)
                elif name.startswith("layers.0"):  # The first layer does not have ReLU applied on its input
                    param.data.normal_(0, 1 / math.sqrt(param.shape[1]))
                else:
                    param.data.normal_(0, math.sqrt(2) / math.sqrt(param.shape[1]))
            elif initialization_method in {"identity"}:
                if name.endswith(".bias"):
                    param.data.fill_(0)
                else:
                    param.data.copy_(torch.eye(param.shape[0]))                
            elif initialization_method in {"user"}:
                if not type(initialization_value) == np.ndarray:
                    raise TypeError(f"Expected np.ndarray for initialization_value; received {type(initialization_value)}")
                if not initialization_value.shape == (param.shape[0],param.shape[0]):
                    raise np.linalg.LinAlgError(f"Shape for initialization_value must be {(param.shape[0],param.shape[0])}. Received {initialization_value.shape}.")
                if name.endswith(".bias"):
                    param.data.fill_(0)
                else:
                    param.data.copy_(torch.tensor(initialization_value))                
            else:
                raise ValueError(f"'initialization_method' may be 'kaiming' or 'identity' or 'user'; received {initialization_method}")
