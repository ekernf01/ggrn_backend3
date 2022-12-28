### GGRN software specifications

From the grammar set out in `GGRN.md`, many of the combinations that can be specified either make no sense, or would be very complex to implement. The current software is has two separate backends with a third planned. Each has a different subset of features.

### Backend 1: basics 

The initial implementation requires the steady-state matching method. It offers flexible regression, rudimentary priors, rudimentary cell type specificity, and predictions after just one step forward. It has no other scheme for matching controls to treatment samples, no acyclic penalty, no low-dimensional structure, and no biological noise. It can be summarized by the following JSON.

    {
        "matching": "steady_state",
        "acyclic_penalty": none,
        "prediction_timescale": "1",
        "perturbation_is_persistent": none,
        "regression_method": ["bunch of sklearn options"],
        "low_dimensional_structure": none,
        "low_dimensional_training": none,
        "sparsity": ["none", "prune_and_refit", "built_in"],
        "known_interactions":    ["none", "hard_threshold"],
        "do_cell_type_specific": false,
        "is_noise_biological": false
    }

Initial benchmarks across the above attributes were entirely negative: the mean of the training data outperforms all other methods. This motivated backend 2.

### Backend 2: DCD-FG 

One of the most promising benchmarks in the literature is on DCD-FG, so we are incorporating DCD-FG into GGRN. This offers a thin wrapper with no features beyond the original DCD-FG implementation. Fortunately, the original DCD-FG implementation spans a variety of related methods. 

    {
        "matching": "steady_state",
        "acyclic_penalty": ["spectral", "exp"],
        "prediction_timescale": "Inf",
        "perturbation_is_persistent": true,
        "regression_method": ["linear", "multilayer_perceptron"],
        "low_dimensional_structure": "EncoderDecoder",
        "low_dimensional_training": "supervised",
        "sparsity": ["built_in"],
        "known_interactions":    ["none"],
        "do_cell_type_specific": false,
        "is_noise_biological": true 
    }

Initial results from modifying the DCD-FG repo show that DCD-FG performs worse than a simple Gaussian baseline that uses no information about interventions (and makes the same prediction for every held-out condition). We will continue to work with DCD-FG, hoping for a believable win over baseline, but we also plan to extend GGRN to explore key elements missing from DCD-FG. 

### Backend 3: autoregressive with matching

We next plan to implement the following features.

    {
        "matching": ["random_control", "closest_control"]
        "acyclic_penalty": none,
        "prediction_timescale": "real",
        "perturbation_is_persistent": true,
        "regression_method": ["linear", "multilayer_perceptron"],
        "low_dimensional_structure": ["none", "QiGQ"],
        "low_dimensional_training": ["supervised", "PCA", "fixed"],
        "sparsity": ["built_in"],
        "known_interactions":    ["eric_is_confused"],
        "do_cell_type_specific": false,
        "is_noise_biological": true 
    }

Formally, we will minimize

$$ L(X) = \sum_{i\in treated} ||(R \circ G \circ Q \circ P_i)^S(X_{M(i)}) - X_i||^2 + \\
\sum_{i\in steady} ||(R \circ G \circ Q\circ P_i)(X_{i}) - X_i||^2 + \\ 
J(G, Q, R) $$

where:

- $treated$ is a set of indices for samples that have undergone treatment.
- $steady$ is a set of samples that is assumed to have reached a steady state (usually, all controls).
- $Q$ is a projection matrix and $R$ is a right-inverse for $Q$. To be clear, we want $z=Q(R(z))$ for lower-dimensional $z$, but we cannot expect $R(Q(x))$ for higher-dimensional $x$. $Q$ and $R$ can be learned by PCA, OR learned by backprop, OR specified by the user as e.g. motif counts + pseudoinverse, OR set to the identity matrix. Maybe eventually we could have some rows fixed by the user and others optimized.
- $G$ predicts a single step forward in time by $T/S$ hours, where $T$ is the duration of treatment.
- $P_i$ enforces interventions on genes perturbed in sample $i$.
- $F^S(X) = F(F(...F(X)))$ ($S$ iterations of $F(X)$).
- $M(i)$ is the index of a control sample matched to treated sample $i$. $M(i)$ can be implemented by choosing a random control, OR by choosing the closest control, OR maybe eventually by choosing $M$ to minimize $L(X)$, OR by optimal transport. 
- $J$ is a regularizer, e.g. an L1 penalty on entries of matrices representing $G$ if $G$ is linear.

This framework can be trained on time-series data, interventional data, or a mixture. This will already be a distinctive advantage whether or not it actually wins benchmarks. For time-series data, $S$ should be adjusted per-sample so it is proportional to the time elapsed. 

This backend has some deliberate omissions. First, it is not clear to me whether and how prior information should be included, but the most likely method is to include motif information the way ARMADA does: by fixing $Q$ and $R$. Second, you can imagine various schemes for sharing information across cell types but we omit that goal from the project scope for now. Third, there is no separation between true and observed expression levels; a true measurement model would require separate quantities for our predictions about the true expression state, similar to the output of a Kalman filter.

We will implement this in Pytorch, so that we can fit any differentiable functions for R, P, G, Q, and J.

#### Pitfalls

- A linear function $F$ is not uniquely determined by equations of the form $y=F^2(x)$, because it can be replaced by $-F$. Because of this, it will be important to always have at least a few training examples where $y=F(x)$ (usually at least the controls, and sometimes all samples if steady state is assumed). This becomes less of an issue if we have irregularly spaced time-points, or if we shrink $F$ towards the identity function (absent evidence, prefer steadiness over oscillation).
- It is probably hard to train this type of model when $S>2$. A possible work-around/relaxation would be to add some intermediates: instead of minimizing $||Y - F^4(X)||^2$, minimize $||Y - H^2(X)||^2 + \lambda||F^2(X) - H(X)||^2$. For linear models, $H=F^2$ for free; for neural nets, $H$ would act as an approximation for $F^2$, but probably shallower. 
- A different work-around (linear only) would be to represent $F$ via its eigendecomposition. 
- Yet another relaxation would be to minimize $||Y - F_1(F_2(X))||^2 + \lambda||F_1(X) - F_2(X)||^2$.
- Another possible pitfall is if $z\neq Q((z))$. Adding $||Q(R(z)) - z||^2$ to $J$ could help, or $R$ could be absorbed into $G$ as in DCD-FG.

#### Software testing

The vanilla version of this is easy to test: generate random controls X and a random transition matrix G; do some perturbations; apply G to them; and feed it into the software. Do not specify the controls as having reached steady state. Manually specify the matching of the treated samples to the right controls. It's also easy to do similar tests while restricting nonzeroes in G to known interactions.

Other features present certain obstacles to testing code correctness. 

- What is the expected result if matched controls are not provided by the user? Maybe if the controls are very far apart, matching each treated sample to the closest control should guarantee correct results. Otherwise, I have no idea.
- If PCA is used to project data into a low dimensional subspace, is there a way of generating the data so that the results are exactly known? I doubt it because generative models corresponding to PCA don't allow interventions. Maybe there is some simple case if you keep all the PC's nicely axis-aligned. 
- If MLP's are used, there are many permutations that would yield equivalent output, so it's hard to guarantee correct results.


#### Optimization notes

As of 2022 Dec 27, the code is in an interesting state where it converges to the right answer on low-D toy examples some of the time, and other times it displays a couple different problems. Sometimes it refuses to converge and other times it is making progress but rapidly blows up to very large weights or biases. Here is what we have begun to experiment with.

- Size and dimension of training data
- Values of S up to 7 (mostly 1 and 2)
- Random seed (after which everything is deterministic)
- Initialization of G & regularization target
- Early stopping

Anecdotally:

- higher S and higher dimension makes training harder. 
- Regular initializations designed for independent layers (e.g. kaiming or xavier) do not work as well for our model, which is structured more like a recurrent neural network. A better default is the identity matrix, used by Geoff Hinton's group in [this RNN initialization paper](https://arxiv.org/pdf/1504.00941.pdf).
- Early stopping is often too early. It's often described as a way to prevent overfitting: you stop when the validation loss increases, even if the training loss could decrease further ([example](https://medium.com/pytorch/pytorch-lightning-1-3-lightning-cli-pytorch-profiler-improved-early-stopping-6e0ffd8deb29)). Right now, I just want to fit the training data really well, but I'm using early stopping because there seems to be no other way to test for convergence when using pytorch lightning.

