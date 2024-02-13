import pandas as pd
from skimage.io import imread

from graph_cl.graph_builder.knn_graph_builder import KNNGraphBuilder
from graph_cl.graph_builder.radius_graph_builder import RadiusGraphBuilder
from graph_cl.graph_builder.contact_graph_builder import ContactGraphBuilder

import yaml
from pathlib import Path

root = Path(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/02_processed/"
)
mask_path = root / "masks" / "BaselTMA_SP41_100_X15Y5.tiff"
mask = imread(mask_path, plugin="tifffile")

with open(
    "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/00_concepts/concept_2_knn.yaml",
    "r",
) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

builder = KNNGraphBuilder()
g = builder.build_graph(mask, **config["graph"]["kwargs"])
