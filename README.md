### Autoregressive models for gene regulatory network inference

This Python package implements flexible vector autoregressive models for gene regulatory network modeling as part of the Grammar of Gene Regulatory Network inference methods (GGRN). For a formal mathematical specification of the objective functions we want to minimize, visit the GGRN repo. 

### Installation

TBD

### Usage

    import ggrn_backend3
    model = ggrn_backend3.GGRNAutoregressiveModel(
        train_data = input_data, 
        matching_method = "user",
        S = 1,
        regression_method = "linear",
        low_dimensional_structure = "none",
    )
    trainer = model.train(
        experiment_name           = "demo",
    )
    result = model.predict()