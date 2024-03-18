from pathlib import Path
from ai4bmr_core.datasets.Dataset import BaseDataset
from pydantic import computed_field


class GCLDataset(BaseDataset):
    @computed_field
    @property
    def concept_graphs_dir(self) -> Path:
        return self.base_dir / "03_concept_graphs"

    @computed_field
    @property
    def samples_dir(self) -> Path:
        return self.base_dir / "04_samples"

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
    def img_dir(self) -> Path:
        return self.processed_dir / "imgs"

    @computed_field
    @property
    def info_path(self) -> Path:
        return self.processed_dir / "info.parquet"

    @computed_field
    @property
    def features_obs_dir(self) -> Path:
        return self.processed_dir / "features" / "observations"

    @computed_field
    @property
    def features_obs_expression_dir(self) -> Path:
        return self.features_obs_dir / "expression"

    @computed_field
    @property
    def features_obs_location_dir(self) -> Path:
        return self.features_obs_dir / "location"

    @computed_field
    @property
    def features_obs_spatial_dir(self) -> Path:
        return self.features_obs_dir / "spatial"

    @computed_field
    @property
    def labels_obs_dir(self) -> Path:
        return self.processed_dir / "labels" / "observations"

    @computed_field
    @property
    def labels_samples_dir(self) -> Path:
        return self.processed_dir / "labels" / "samples"

    @computed_field
    @property
    def metadata_samples_dir(self) -> Path:
        return self.metadata_dir / "samples"

    def get_sample_paths(self, sample_name) -> dict[str, Path]:
        return {
            # features
            "expression_url": self.features_obs_expression_dir
            / f"{sample_name}.parquet",
            "location_url": self.features_obs_location_dir / f"{sample_name}.parquet",
            "spatial_url": self.features_obs_spatial_dir / f"{sample_name}.parquet",
            # labels
            "labels_url": self.labels_obs_dir / f"{sample_name}.parquet",
            "labels_sample_url": self.labels_samples_dir / f"{sample_name}.parquet",
            # other
            "mask_url": self.mask_dir / f"{sample_name}.tiff",
            "img_url": self.img_dir / f"{sample_name}.tiff",
            "metadata_url": self.metadata_samples_dir / f"{sample_name}.parquet",
        }

    def get_concept_graph_path(self, concept_name: str, sample_name: str) -> Path:
        return self.concept_graphs_dir / concept_name / f"{sample_name}.pt"

    def get_sample_path_by_name(self, sample_name: str):
        return self.samples_dir / f"{sample_name}.json"
