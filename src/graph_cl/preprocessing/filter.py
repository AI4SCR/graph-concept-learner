import numpy as np
import pandas as pd
from ai4bmr_core.log.log import logger

from ..data_models.Sample import Sample


def filter_has_mask(samples: list[Sample]) -> list[Sample]:
    def has_mask(sample: Sample) -> bool:
        eval = sample.mask_url is not None
        if not eval:
            logger.debug(f"{sample.name} filtered out because it does not a mask 🎭.")
        return eval

    return list(filter(has_mask, samples))


def filter_has_target(samples: list[Sample]) -> list[Sample]:
    def has_target(sample: Sample) -> bool:
        eval = pd.notna(sample.target)
        if not eval:
            logger.debug(
                f"{sample.name} filtered  out because it does not have a target 🎯."
            )
        return eval

    return list(filter(has_target, samples))


def filter_has_enough_nodes(
    samples: list[Sample], concept_names: list[str], min_num_nodes: int
) -> list[Sample]:
    def has_enough_nodes(sample: Sample) -> bool:
        for concept_name in concept_names:
            if sample.get_concept_graph(concept_name) is None:
                logger.debug(
                    f"{sample.name} does not have concept {concept_name}. Did you run `graph_cl experiment create-concept-graph`?"
                )
                return False

            eval = sample.get_concept_graph(concept_name).num_nodes > min_num_nodes
            if not eval:
                logger.debug(
                    f"{sample.name} filtered  out because concept {concept_name} does not have {min_num_nodes} nodes."
                )
                return False
        return True

    return list(filter(has_enough_nodes, samples))


def filter_has_labels(samples: list[Sample]) -> list[Sample]:
    def has_mask_and_labels(sample: Sample) -> bool:
        eval = sample.labels_url is not None or sample.labels_sample_url
        if not eval:
            logger.debug(
                f"{sample.name} filtered out because it does not have labels 🏷️ ."
            )
        return eval

    return list(filter(has_mask_and_labels, samples))


def filter_mask_objects(mask: np.ndarray, labels: pd.Series, include_labels: list):
    m = labels.isin(include_labels)
    obj_ids = set(labels[m].index.get_level_values("cell_id"))
    mask_objs = set(mask.flatten())
    mask_objs.remove(0)

    for obj in mask_objs - obj_ids:
        mask[mask == obj] = 0

    return mask
