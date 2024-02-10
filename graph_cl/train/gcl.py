# Load concept GNN models
model_dict = {}  # Init model dictionary

# Load models
for concept, data_model_checkpoints in concept_set_cfg.items():
    # Unpack paths
    path_to_model_config = concept_set_cfg[concept]["config"]
    path_to_model_checkpoint = concept_set_cfg[concept]["checkpoint"]

    # Since we are already looping this dict save it to the gcl_cfg s.t. it is logged to mlflow
    gcl_cfg[f"{concept}.path_to_gnn_config"] = path_to_model_config
    gcl_cfg[f"{concept}.path_to_gnn_checkpoint"] = path_to_model_checkpoint

    # Load it
    with open(path_to_model_config) as file:
        concept_cfg = yaml.load(file, Loader=yaml.Loader)

    # Load dataset
    concept_dataset = Concept_Dataset(dataset.concept_dict[concept])

    # Get concept training dataset (needed it to instantiate PNA GNN models)
    concept_splitted_dataset = split_concept_dataset(
        splits_df=splits_df, index_col="core", dataset=concept_dataset
    )

    # Get number of classes
    concept_cfg["num_classes"] = concept_dataset.num_classes
    concept_cfg["in_channels"] = concept_dataset.num_node_features
    concept_cfg["hidden_channels"] = concept_cfg["in_channels"] * concept_cfg["scaler"]

    # Load model
    model = GNN_plus_MPL(concept_cfg, concept_splitted_dataset["train"])

    # Load checkpoint
    if "end_to_end" not in gcl_cfg.keys():
        model.load_state_dict(torch.load(path_to_model_checkpoint, map_location=device))
    elif gcl_cfg["end_to_end"]:
        pass
    else:
        model.load_state_dict(torch.load(path_to_model_checkpoint, map_location=device))

    # Remove head
    model = model.get_submodule("gnn")

    # Add to dictionary
    model_dict[concept] = model

# Check if all models have the same output dimension
out_dims = np.array([])
for concept, model in model_dict.items():
    out_dims = np.append(out_dims, model.gnn.out_channels)
assert all(
    out_dims == out_dims[0]
), "Not all graph embeddings for the different concept learners are the same dimension."
