### Autoregressive models for gene regulatory network inference

This Python package implements flexible vector autoregressive models for gene regulatory network modeling as part of the Grammar of Gene Regulatory Network inference methods (GGRN). For a formal mathematical specification of the objective functions we want to minimize, visit the GGRN repo. In brief, we fit models assuming a expression profile $X_t$ measured $S$ time-steps after perturbing gene $P$ can be matched to a control $X_c$ such that $X_t = F^S(X_c, P)$ with some strong assumptions about $F$.

### Installation

TBD

### Usage

```python
import ggrn_backend3.api as ggrn_backend3
linear_autoregressive, R,G,Q,F, factors = ggrn_backend3.simulate_autoregressive(num_controls_per_group=10, num_features = 3)
model = ggrn_backend3.GGRNAutoregressiveModel(
    linear_autoregressive, 
    matching_method = "user",     # Future plans: "random", "closest"
    regression_method = "linear", # Future plans: "multilayer_perceptron"
    low_dimensional_structure = "RGQ", #or "none"
    low_dimensional_training = "supervised", # or "SVD" or "fixed"
    low_dimensional_value = 1, #latent dimension or, if low_dimensional_training=="fixed", an entire projection matrix 
    S=1, # any positive int
    network = None, # use a LightNetwork such as example_network if low_dimensional_training=="fixed"
)
model.train()
# For a simulation without genetic perturbation: use "control" as the gene name.
y = model.predict(perturbations = [("control", 0)], starting_expression = linear_autoregressive[0,:])
# Single-gene perturbation
y = model.predict(perturbations = [("1", 0)], starting_expression = linear_autoregressive[0,:])
# Two single-gene perturbations of separate cells
y = model.predict(perturbations = [("1", 0), ("1", 0)], starting_expression = linear_autoregressive[0,:])
# One sample with two genes perturbed
y = model.predict(perturbations = [("1,2", "0,0")], starting_expression = linear_autoregressive[0,:])
# The following perturbation formats will NOT work. 
[
    ("control,2", "0,0")  # Not allowed
    ("notagene", 0)  # Not allowed
]
```