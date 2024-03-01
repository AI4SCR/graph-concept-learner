import logging

from skimage.io import imread
import pandas as pd
import yaml
from pathlib import Path
import torch

from graph_cl.graph_builder.build_graph import build_pyg_graph
from graph_cl.preprocessing.filter import filter_mask_objects
from graph_cl.configuration.configurator import ConceptConfig
import click


@click.command()
@click.argument("mask_path", type=click.Path(exists=True, path_type=Path))
@click.argument("labels_path", type=click.Path(exists=True, path_type=Path))
@click.argument("concept_config_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--force", is_flag=True, default=False, help="Force overwrite of existing graph."
)
def build_concept_graph(
    mask_path: Path,
    labels_path: Path,
    concept_config_path: Path,
    output_path: Path,
    force: bool = False,
):
    logging.info(f"Building graph for {mask_path.stem}")

    with open(concept_config_path) as f:
        concept_config = yaml.load(f, Loader=yaml.Loader)
        concept_config = ConceptConfig(**concept_config)

    # core = mask_path.stem
    # concept_name = concept_config.concept_name
    # output_path = output_dir / concept_name / f"{core}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not force and output_path.exists():
        logging.info(f"Graph for {mask_path.stem} already exists.")
        return

    labels = pd.read_parquet(labels_path)
    labels = labels[concept_config.filter.col_name]

    mask = imread(mask_path, plugin="tifffile")
    mask = filter_mask_objects(
        mask, labels=labels, include_labels=concept_config.filter.include_labels
    )
    graph = build_pyg_graph(
        mask=mask,
        topology=concept_config.graph.topology,
        params=concept_config.graph.params,
    )

    torch.save(graph, output_path)
