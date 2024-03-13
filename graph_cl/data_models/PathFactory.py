from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, computed_field
from dotenv import find_dotenv


class PathFactory(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(".env", usecwd=True),
        protected_namespaces=("settings_",),
        extra="ignore",
    )

    data_dir: Path
    dataset_name: None | str = None

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
            # "attributes_url": self.attributes_dir / f"{sample_name}.parquet",
            "metadata_url": self.metadata_dir / "samples" / f"{sample_name}.parquet",
            "sample_labels_url": self.processed_dir
            / "labels"
            / "samples"
            / f"{sample_name}.parquet",
        }

    def get_concept_path(self, concept_name) -> None | Path:
        if self.concepts_dir is None:
            return None
        return self.concepts_dir / f"{concept_name}.yaml"

    def get_concept_graph_path(
        self, concept_name: str, sample_name: str
    ) -> None | Path:
        if self.concept_graphs_dir is None:
            return None
        return self.concept_graphs_dir / concept_name / f"{sample_name}.pt"

    @computed_field
    @property
    def dataset_dir(self) -> None | Path:
        if self.dataset_name is None:
            return None
        return self.data_dir / "datasets" / self.dataset_name

    @computed_field
    def concepts_dir(self) -> Path:
        return self.data_dir / "concepts"

    @computed_field
    def experiments_dir(self) -> Path:
        return self.data_dir / "experiments"

    @computed_field
    def concept_graphs_dir(self) -> None | Path:
        if self.dataset_dir is None:
            return None
        return self.dataset_dir / "04_concept_graphs"

    @computed_field
    def raw_dir(self) -> None | Path:
        if self.dataset_dir is None:
            return None
        return self.dataset_dir / "01_raw"

    @computed_field
    def processed_dir(self) -> None | Path:
        if self.dataset_dir is None:
            return None
        return self.dataset_dir / "02_processed"

    @computed_field
    def samples_dir(self) -> None | Path:
        if self.dataset_dir is None:
            return None
        return self.dataset_dir / "03_samples"

    @computed_field
    @property
    def metadata_dir(self) -> None | Path:
        if self.processed_dir is None:
            return None
        return self.processed_dir / "metadata"

    @computed_field
    @property
    def mask_dir(self) -> None | Path:
        if self.processed_dir is None:
            return None
        return self.processed_dir / "masks"

    @field_validator("data_dir")
    @classmethod
    def to_path(cls, v: str | Path) -> Path:
        return Path(v)
