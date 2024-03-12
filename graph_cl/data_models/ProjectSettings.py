from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator, computed_field
from dotenv import find_dotenv


class ProjectSettings(BaseSettings):
    model_config = ConfigDict(
        env_file=find_dotenv(".env", usecwd=True),
        protected_namespaces=("settings_",),
        extra="ignore",
    )
    data_dir: Path
    dataset_name: None | str = None
    experiment_name: None | str = None
    concept_name: None | str = None
    model_name: None | str = None
    sample_name: None | str = None

    def init(self):
        dirs = self.get_directories()
        for d in dirs:
            path = getattr(self, d)
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_directories(cls):
        return [i for i in cls.__dict__ if i.endswith("_dir")]

    @computed_field
    @property
    def dataset_dir(self) -> None | Path:
        if self.dataset_name is None:
            return None
        return self.data_dir / "datasets" / self.dataset_name

    @computed_field
    def concepts_dir(self) -> Path:
        return self.data_dir / "concepts"

    def get_concept_path(self, concept_name) -> None | Path:
        if self.concepts_dir is None:
            return None
        return self.concepts_dir / f"{concept_name}.yaml"

    @computed_field
    def experiments_dir(self) -> Path:
        return self.data_dir / "experiments"

    @computed_field
    def experiment_dir(self) -> None | Path:
        if self.experiment_name is None or self.experiments_dir is None:
            return None
        return self.experiments_dir / self.experiment_name

    @computed_field
    def split_info_path(self) -> None | Path:
        if self.experiment_dir is None:
            return None
        return self.experiment_dir / "split_info.parquet"

    @computed_field
    def experiment_config_dir(self) -> None | Path:
        if self.experiment_dir is None:
            return None
        return self.experiment_dir / "configuration"

    @computed_field
    def data_config_path(self) -> None | Path:
        if self.experiment_dir is None:
            return None
        return self.experiment_config_dir / "data.yaml"

    @computed_field
    def model_gcl_config_path(self) -> None | Path:
        if self.experiment_dir is None:
            return None
        return self.experiment_config_dir / "model_gcl.yaml"

    @computed_field
    def model_gnn_config_path(self) -> None | Path:
        if self.experiment_dir is None:
            return None
        return self.experiment_config_dir / "model_gnn.yaml"

    @computed_field
    def pretrain_config_path(self) -> None | Path:
        if self.experiment_dir is None:
            return None
        return self.experiment_config_dir / "pretrain.yaml"

    @computed_field
    def train_config_path(self) -> None | Path:
        if self.experiment_dir is None:
            return None
        return self.experiment_config_dir / "train.yaml"

    @computed_field
    def raw_dir(self) -> None | Path:
        if self.dataset_dir is None:
            return None
        return self.dataset_dir / "01_raw"

    @computed_field
    def processed_dir(self) -> Path:
        if self.dataset_dir is None:
            return None
        return self.dataset_dir / "02_processed"

    @computed_field
    def samples_dir(self) -> None | Path:
        if self.dataset_dir is None:
            return None
        return self.dataset_dir / "03_samples"

    def get_sample_path(self, sample_name) -> None | Path:
        if self.samples_dir is None:
            return None
        return self.samples_dir / f"{sample_name}.pkl"

    @computed_field
    def concept_graphs_dir(self) -> None | Path:
        if self.dataset_dir is None:
            return None
        return self.dataset_dir / "04_concept_graphs"

    def get_concept_graph_path(
        self, concept_name: str, sample_name: str
    ) -> None | Path:
        if self.concept_graphs_dir is None:
            return None
        return self.concept_graphs_dir / concept_name / f"{sample_name}.pt"

    @computed_field
    def attributes_dir(self) -> None | Path:
        if self.experiment_dir is None:
            return None
        return self.dataset_dir / "05_attributes"

    @computed_field
    def prediction_dir(self) -> None | Path:
        if self.dataset_name is None or self.model_name is None:
            return None
        return self.data_dir / "predictions" / self.dataset_name / self.model_name

    @computed_field
    def prediction_path(self) -> None | Path:
        if self.prediction_dir is None or self.sample_name is None:
            return None
        return self.prediction_dir / self.sample_name

    @computed_field
    def model_dir(self) -> None | Path:
        if self.model_name is None:
            return None
        return self.data_dir / "models" / self.model_name

    @computed_field()
    def result_dir(self) -> None | Path:
        if self.model_name is None or self.dataset_name is None:
            return None
        return self.data_dir / "results" / self.model_name / self.dataset_name

    @computed_field()
    def result_path(self) -> None | Path:
        if self.result_dir is None or self.sample_name is None:
            return None
        return self.result_dir / self.sample_name

    @field_validator("data_dir")
    @classmethod
    def to_path(cls, v: str | Path) -> Path:
        return Path(v)
