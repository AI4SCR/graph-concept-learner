# %%
import athena as ath
import yaml
import pickle

# %%
# (
# prog_name,
# spl,
# cfg_file,
# so_file,
# output_file,
# ) = sys.argv

# %%
def generate_graph(spl, cfg_file, so_file, output_file):

    # Load so obj
    with open(so_file, "rb") as f:
        so = pickle.load(f)

    # Load config
    with open(cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # Generate graphs for config
    concept_name = cfg["concept_name"]
    builder_type = cfg["builder_type"]

    randomize = cfg.pop("randomize", {})
    randomize_cell_labels = randomize.pop("cell_labels", False)
    if randomize_cell_labels:
        randomize_seed = randomize.pop("seed", 42)
        col_name = cfg["concept_params"]["filter_col"]
        so.obs[spl][col_name] = so.obs[spl][col_name].sample(
            frac=1, replace=False, random_state=randomize_seed
        )

    # Do not attribute
    cfg["build_and_attribute"] = False

    # Make a graph for the given sample, and turn it into pyg and save to file
    # Build graph
    ath.graph.build_graph(so, spl, config=cfg, key_added=concept_name)

    # Remove edge weights
    for (n1, n2, d) in so.G[spl][concept_name].edges(data=True):
        d.clear()

    # Write to output
    with open(output_file, "wb") as f:
        pickle.dump(so.G[spl][concept_name], f)


def generate_graphs():
    from ..configuration import CONFIG

    cfg_dir = CONFIG.experiment.root / "configuration" / "concept_configs"
    so_file = CONFIG.data.intermediate
    spl, cfg_file, so_file, output_file
