from athena.utils.default_configs import get_default_config
import os.path as osp
from ruamel import yaml
import argparse

# %% Parse command line arguments to pass to the script
parser = argparse.ArgumentParser(
    description="Base config generateor.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("cfg", help="Path to folder where to dump base config.")
args = parser.parse_args()
args = vars(args)
root = args["cfg"]

c1 = {
    "concept_name": "immune_tumor_radius",
    "graph_type": "radius",
    "build_concept_graph": True,
    "attrs_type": "so",
    "filter_col": "cell_type",
    "labels": ["immune", "tumor"],
}

c2 = {
    "concept_name": "immune_tumor_contact",
    "graph_type": "contact",
    "build_concept_graph": True,
    "attrs_type": "so",
    "filter_col": "cell_type",
    "labels": ["immune", "tumor"],
}

c3 = {
    "concept_name": "immune_tumor_knn",
    "graph_type": "knn",
    "build_concept_graph": True,
    "attrs_type": "so",
    "filter_col": "cell_type",
    "labels": ["immune", "tumor"],
}

# Put concepts into a list
concepts = [c1, c2, c3]

# Make configs for evey concept and write them to root
for concept in concepts:
    # Init empty dict and add concept name so it starts at the top
    config = {}
    config["concept_name"] = concept["concept_name"]

    # Get default config for concept and modify
    def_config = get_default_config(
        builder_type=concept["graph_type"],
        build_concept_graph=concept["build_concept_graph"],
        build_and_attribute=True,  # Must always have attributes.
        attrs_type=concept["attrs_type"],
    )
    def_config["concept_params"]["filter_col"] = concept["filter_col"]
    def_config["concept_params"]["labels"] = concept["labels"]

    # Join the two dicts
    config = {**config, **def_config}

    # Write to file
    file_name = osp.join(root, f'{concept["graph_type"]}_dataset_base_config.yaml')

    with open(file_name, "w") as file:
        documents = yaml.dump(config, file, Dumper=yaml.RoundTripDumper)

    # Done
    print(f"Generated dataset base config and save it to: {file_name}")
