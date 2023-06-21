## Autoregressive models for gene regulatory network inference

This Python package implements flexible vector autoregressive models for gene regulatory network modeling as part of the Grammar of Gene Regulatory Network inference methods (GGRN). In brief, we fit models assuming a expression profile $X_t$ measured $S$ time-steps after perturbing gene $P$ can be matched to a control $X_c$ such that $X_t = F^S(X_c, P)$. We currently assume linear $F$. More details are given below.

### Installation

TBD

### Usage

```python
import ggrn_backend3.api as ggrn_backend3
linear_autoregressive_data, R,G,Q,F, factors = ggrn_backend3.simulate_autoregressive(num_controls_per_group=10, num_features = 3)
model = ggrn_backend3.GGRNAutoregressiveModel(
    linear_autoregressive_data, 
    matching_method = "closest",    
    regression_method = "linear",
    low_dimensional_structure = "dynamics",
    low_dimensional_training = "supervised", 
    low_dimensional_value = 1,  
    S=1
)
model.train()
# For a forward simulation without genetic perturbation: use "control" as the gene name.
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

### Mathematical specification

Formally, this package minimizes

$$ L(X) = \sum_{i\in treated} ||(R \circ G \circ Q \circ P_i)^S(X_{M(i)}) - X_i|| + \\
\sum_{i\in steady} ||(R \circ G \circ Q\circ P_i)(X_{i}) - X_i|| + \\ 
J(G, Q, R) $$

where:

- $treated$ is a set of indices for samples that have undergone treatment.
- $steady$ is a set of samples that is assumed to have reached a steady state (usually, all controls).
- $Q$ is an encoder and $R$ is a right-inverse for $Q$. To be clear, we want $z=Q(R(z))$ for all lower-dimensional $z$, but we cannot expect $R(Q(x))$ for all higher-dimensional $x$. Currently $Q$ and $R$ can be learned by PCA (similer to PRESCIENT), OR learned by backprop (similar to DCD-FG), OR specified by the user as e.g. motif counts + pseudoinverse (similar to ARMADA), OR set to the identity matrix. Maybe eventually we could have some rows fixed by the user and others optimized, as in [PEER](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000770).
- $G$ predicts a single step forward in time by $T/S$ hours, where $T$ is the duration of treatment.
- $P_i$ enforces interventions on genes perturbed in sample $i$ by setting them to a user-specified value.
- $F^S(X) = F(F(...F(X))...)$ ($S$ iterations of $F(X)$).
- $M(i)$ is the index of a control sample matched to treated sample $i$. $M(i)$ can be implemented by choosing a random control, OR by choosing the closest control, or can be user-provided. 
- $J$ is a regularizer. We use an L1 penalty on entries of matrices representing $G$ if $G$ is linear.

This framework can in theory be trained on time-series data, interventional data, or a mixture. For time-series data, $S$ should be adjusted per-sample so it is proportional to the time elapsed. So far, time-series handling is not implemented; we focus on perturbation experiments where all outcomes are measured at the same time-point. 