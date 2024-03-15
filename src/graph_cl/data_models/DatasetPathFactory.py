from pathlib import Path
from pydantic import computed_field

from .ProjectPathFactory import ProjectPathFactory


class DatasetPathFactory(ProjectPathFactory):
    dataset_name: Path | str

    @computed_field
    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / "datasets" / self.dataset_name

    @computed_field
    @property
    def raw_dir(self) -> Path:
        return self.dataset_dir / "01_raw"

    @computed_field
    @property
    def processed_dir(self) -> Path:
        return self.dataset_dir / "02_processed"

    @computed_field
    @property
    def concept_graphs_dir(self) -> Path:
        return self.dataset_dir / "03_concept_graphs"

    @computed_field
    @property
    def samples_dir(self) -> Path:
        return self.dataset_dir / "04_samples"

    @computed_field
    @property
    def processed_samples_dir(self) -> Path:
        return self.processed_dir / "samples"

    @computed_field
    @property
    def metadata_dir(self) -> Path:
        return self.processed_dir / "metadata"

    @computed_field
    @property
    def mask_dir(self) -> Path:
        return self.processed_dir / "masks"

    @computed_field
    @property
    def info_path(self) -> Path:
        return self.processed_dir / "info.parquet"

    def get_sample_paths(
        self, sample_name, concept_names: list[str] = []
    ) -> dict[str, Path]:
        feat_obs_dir = self.processed_dir / "features" / "observations"
        return {
            "expression_url": feat_obs_dir / "expression" / f"{sample_name}.parquet",
            "location_url": feat_obs_dir / "location" / f"{sample_name}.parquet",
            "spatial_url": feat_obs_dir / "spatial" / f"{sample_name}.parquet",
            "labels_url": self.processed_dir
            / "labels"
            / "observations"
            / f"{sample_name}.parquet",
            "mask_url": self.mask_dir / f"{sample_name}.tiff",
            "img_url": self.processed_dir / "imgs" / f"{sample_name}.tiff",
            # "concept_graph_url": {concept_name: self.get_concept_graph_path(concept_name, sample_name)
            #                       for concept_name in concept_names},
            "metadata_url": self.metadata_dir / "samples" / f"{sample_name}.parquet",
            "sample_labels_url": self.processed_dir
            / "labels"
            / "samples"
            / f"{sample_name}.parquet",
        }

    def get_concept_graph_path(self, concept_name: str, sample_name: str) -> Path:
        return self.concept_graphs_dir / concept_name / f"{sample_name}.pt"
