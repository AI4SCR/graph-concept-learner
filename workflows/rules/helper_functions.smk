##### Helper functions #####
def get_concept_sets(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/attribute_configs"
    ATTR_CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if f.endswith(".yaml")]

    return expand(
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/train_configs/concept_sets/"+"{labels_permuted}/{attribute_config}/concept_set.yaml",
        labels_permuted=["not_permuted"],
        attribute_config=ATTR_CFG_IDS
    )

def get_configs_best_pretrain_models(wildcards):
    path_to_configs = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/attribute_configs"
    CONFIG_NAMES = [os.path.splitext(f)[0] for f in os.listdir(path_to_configs) if f.endswith(".yaml")]

    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/concept_configs"
    CONCEPT_NAMES = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if f.endswith(".yaml")]

    return expand(
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_model_per_concept/{{labels_permuted}}/{{attribute_config}}/{{concept}}.yaml",
        concept=CONCEPT_NAMES,
        attribute_config=CONFIG_NAMES,
        labels_permuted=["not_permuted"]
    )

def get_all_attributed_graphs(wildcards):
    path_to_configs = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/attribute_configs"
    CONFIG_NAMES = [os.path.splitext(f)[0] for f in os.listdir(path_to_configs) if f.endswith(".yaml")]

    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/concept_configs"
    CONCEPT_NAMES = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if f.endswith(".yaml")]

    path_to_folds = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/normalized_data"
    FOLDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_folds) if f.endswith(".pkl")]

    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/processed_data/attributed/{{attribute_config}}/{{concept}}/{{fold}}", concept=CONCEPT_NAMES, attribute_config=CONFIG_NAMES, fold=FOLDS)

def get_all_graphs_and_datasets(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/concept_configs"
    CONCEPT_NAMES = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if f.endswith(".yaml")]

    CONTACT_CONCEPTS = []
    RADIUS_KNN_CONCEPTS = []
    for concept in CONCEPT_NAMES:
        if "contact" in concept:
            CONTACT_CONCEPTS.append(concept)
        else:
            RADIUS_KNN_CONCEPTS.append(concept)

    path_to_filtered_sample_ids = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/filtered_sample_ids_and_labels.csv"
    SMAPLE_IDS = list(pd.read_csv(path_to_filtered_sample_ids, index_col=0).squeeze("columns").index.values)

    all_graphs = expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/processed_data/unattributed/{{concept}}/{{spl_id}}.pkl", concept=CONTACT_CONCEPTS, spl_id=SMAPLE_IDS)
    all_datasets = expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/processed_data/unattributed/{{concept}}", concept=RADIUS_KNN_CONCEPTS)

    return all_graphs + all_datasets

def get_all_graphs_for_a_concept(wildcards):
    path_to_filtered_sample_ids = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/filtered_sample_ids_and_labels.csv"
    SMAPLE_IDS = list(pd.read_csv(path_to_filtered_sample_ids, index_col=0).squeeze("columns").index.values)
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/processed_data/" + "{{concept}}/{spl_id}.pt", spl_id=SMAPLE_IDS)

def get_paths_to_pretrained_models(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/concept_configs"
    CONCEPT_NAMES = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if f.endswith(".yaml")]

    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/pretrain_model_configs"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if f.endswith(".yaml")]

    FOLD_IDS = [f"fold_{i}" for i in range(config["n_folds"])]
    SEEDS = [f"seed_{i}" for i in range(config["n_seeds"])]

    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/attribute_configs"
    ATTR_CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if f.endswith(".yaml")]

    return expand(
        f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/{{labels_permuted}}/{{attribute_config}}/{{concept}}/{{cfg_id}}/{{fold}}/{{seed}}/best_val_{{metric_name}}.pt",
        fold=FOLD_IDS,
        seed=SEEDS,
        concept=CONCEPT_NAMES,
        cfg_id=CFG_IDS,
        attribute_config=ATTR_CFG_IDS,
        metric_name=config["follow_this_metrics"],
        labels_permuted=["not_permuted"]
    )

def get_paths_to_concept_configs(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/concept_configs"
    CONCEPT_NAMES = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{path_to_file}/{{concept}}.yaml", concept=CONCEPT_NAMES)

def gen_paths_to_train_conf_matrices(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/train_model_configs"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/graph_concept_learners/"+"ER/{config_id}/test_conf_mat_from_best_val_{metric_name}.png", metric_name=config["follow_this_metrics"], config_id=CFG_IDS)

def gen_paths_to_space_gm_conf_matrices(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/space_gm_train_configs"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    path_to_file = f"{root}/intermediate_data/non_spatial_baseline/"
    DATASET_NAMES = [f for f in os.listdir(path_to_file)]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/space_gm_baseline/"+"{dataset}/{cfg_id}/test_conf_mat_from_best_val_weighted_f1_score.png", cfg_id=CFG_IDS, dataset=DATASET_NAMES)

def get_all_rand_graphs_and_datasets(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/concept_configs"
    CONCEPT_NAMES = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]

    CONTACT_CONCEPTS = []
    RADIUS_CONCEPTS = []
    for concept in CONCEPT_NAMES:
        if "contact" in concept:
            CONTACT_CONCEPTS.append(concept)
        else:
            RADIUS_CONCEPTS.append(concept)

    path_to_filtered_sample_ids = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/filtered_sample_ids_and_labels.csv"
    SMAPLE_IDS = list(pd.read_csv(path_to_filtered_sample_ids, index_col=0).squeeze("columns").index.values)

    all_graphs = expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/randomized_data/{{concept}}/{{spl_id}}.pt", concept=CONTACT_CONCEPTS, spl_id=SMAPLE_IDS)
    all_datasets = expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/randomized_data/{{concept}}", concept=RADIUS_CONCEPTS)

    return all_graphs + all_datasets

### Helpers for random ER concepts ###
def get_rnd_all_cells_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_all_cells_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/randomized_data/all_cells_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_all_cells_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_all_cells_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/randomized_data/all_cells_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_endothelial_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_endothelial_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/randomized_data/endothelial_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_endothelial_stromal_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_endothelial_stromal_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/randomized_data/endothelial_stromal_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_endothelial_tumor_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_endothelial_tumor_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/randomized_data/endothelial_tumor_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_immune_endothelial_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_immune_endothelial_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/randomized_data/immune_endothelial_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_immune_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_immune_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/randomized_data/immune_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_immune_stromal_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_immune_stromal_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/randomized_data/immune_stromal_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_immune_tumor_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_immune_tumor_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/randomized_data/immune_tumor_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_stromal_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_stromal_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/randomized_data/stromal_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_stromal_tumor_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_stromal_tumor_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/randomized_data/stromal_tumor_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_tumor_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_tumor_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/randomized_data/tumor_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

### Helpers for seed concepe pretraining ###
def get_best_spaceGM_ER(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_spaceGM_ER"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/space_gm_baseline/"+"best_spaceGM_ER/frequency_dataset/{cfg_id}/test_conf_mat_from_best_val_weighted_f1_score.png", cfg_id=CFG_IDS)

def get_best_spaceGM_ERless(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_spaceGM_ERless"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/space_gm_baseline/"+"best_spaceGM_ERless/frequency_dataset_ERless/{cfg_id}/test_conf_mat_from_best_val_weighted_f1_score.png", cfg_id=CFG_IDS)

def get_best_all_cells_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_all_cells_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/all_cells_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_all_cells_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_all_cells_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/all_cells_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_all_cells_ERless_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_all_cells_ERless_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/all_cells_ERless_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_all_cells_ERless_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_all_cells_ERless_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/all_cells_ERless_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_endothelial_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_endothelial_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/endothelial_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_endothelial_ERless_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_endothelial_ERless_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/endothelial_ERless_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_endothelial_stromal_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_endothelial_stromal_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/endothelial_stromal_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_endothelial_stromal_ERless_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_endothelial_stromal_ERless_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/endothelial_stromal_ERless_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_endothelial_tumor_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_endothelial_tumor_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/endothelial_tumor_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_endothelial_tumor_ERless_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_endothelial_tumor_ERless_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/endothelial_tumor_ERless_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_immune_endothelial_ERless_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_immune_endothelial_ERless_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/immune_endothelial_ERless_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_immune_endothelial_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_immune_endothelial_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/immune_endothelial_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_immune_ERless_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_immune_ERless_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/immune_ERless_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_immune_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_immune_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/immune_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_immune_stromal_ERless_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_immune_stromal_ERless_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/immune_stromal_ERless_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_immune_stromal_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_immune_stromal_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/immune_stromal_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_immune_tumor_ERless_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_immune_tumor_ERless_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/immune_tumor_ERless_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_immune_tumor_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_immune_tumor_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/immune_tumor_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_stromal_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_stromal_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/stromal_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_stromal_ERless_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_stromal_ERless_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/stromal_ERless_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_stromal_tumor_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_stromal_tumor_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/stromal_tumor_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_stromal_tumor_ERless_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_stromal_tumor_ERless_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/stromal_tumor_ERless_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_tumor_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_tumor_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/tumor_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_tumor_ERless_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_tumor_ERless_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/tumor_ERless_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

### Helpers for seed gcl training ###
def get_best_gcl_ER_linear(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_gcl_ER_linear"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/graph_concept_learners/"+"best_gcl_ER_linear/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_gcl_ER_concat(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_gcl_ER_concat"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/graph_concept_learners/"+"best_gcl_ER_concat/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_gcl_ER_transformer(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_gcl_ER_transformer"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/graph_concept_learners/"+"best_gcl_ER_transformer/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_gcl_ERless_linear(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_gcl_ERless_linear"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/graph_concept_learners/"+"best_gcl_ERless_linear/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_gcl_ERless_concat(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_gcl_ERless_concat"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/graph_concept_learners/"+"best_gcl_ERless_concat/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_gcl_ERless_transformer(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_gcl_ERless_transformer"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/graph_concept_learners/"+"best_gcl_ERless_transformer/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

### Helper functions for randomly permuted n seed gcl training ###
def get_rnd_gcl_ER_linear(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_gcl_ER_linear"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/graph_concept_learners/randomized_data/"+"best_gcl_ER_linear/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_gcl_ER_concat(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_gcl_ER_concat"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/graph_concept_learners/randomized_data/"+"best_gcl_ER_concat/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_gcl_ER_transformer(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/best_gcl_ER_transformer"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/graph_concept_learners/randomized_data/"+"best_gcl_ER_transformer/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

### Rules fro end 2 end configs ###
def get_end_2_end_conf_mats(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs/end_2_end_configs"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/checkpoints/graph_concept_learners/end_2_end/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

### rule to normalize all folds ###
def get_normalize_all_folds(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/CV_folds/folds/"
    FOLD_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".csv"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/meta_data/normalized_data/"+"{fold}.pkl", fold=FOLD_IDS)
