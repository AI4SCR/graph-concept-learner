from pathlib import Path
from pydantic import computed_field
from .ProjectPathFactory import ProjectPathFactory


class ExperimentPathFactory(ProjectPathFactory):
    experiment_name: Path | str

    # TODO: would it make sense to allow to load those from the .env?
    # model_name: None | str = None
    # dataset_name: None | str = None

    @computed_field
    @property
    def experiment_dir(self) -> Path:
        return self.experiments_dir / self.experiment_name

    @computed_field
    @property
    def split_info_path(self) -> Path:
        return self.experiment_dir / "split_info.parquet"

    @computed_field
    @property
    def config_dir(self) -> Path:
        return self.experiment_dir / "00_configuration"

    @computed_field
    @property
    def attributes_dir(self) -> Path:
        return self.experiment_dir / "05_attributes"

    @computed_field
    @property
    def experiment_samples_dir(self) -> Path:
        return self.experiment_dir / "06_samples"

    @computed_field
    @property
    def experiment_dataset_dir(self) -> Path:
        return self.experiment_dir / "07_datasets"

    @computed_field
    @property
    def model_dir(self) -> Path:
        return self.experiment_dir / "08_models"

    @computed_field
    @property
    def model_gcl_dir(self) -> Path:
        return self.experiment_dir / "08_models" / "gcl"

    @computed_field
    @property
    def model_gnn_dir(self) -> Path:
        return self.experiment_dir / "08_models" / "gnn"

    @computed_field
    @property
    def prediction_dir(self) -> Path:
        return self.experiment_dir / "09_predictions"

    @computed_field
    @property
    def result_dir(self) -> Path:
        return self.experiment_dir / "10_results"

    @computed_field
    @property
    def data_config_path(self) -> Path:
        return self.config_dir / "data.yaml"

    @computed_field
    @property
    def model_gcl_config_path(self) -> Path:
        return self.config_dir / "model_gcl.yaml"

    @computed_field
    @property
    def model_gnn_config_path(self) -> Path:
        return self.config_dir / "model_gnn.yaml"

    @computed_field
    @property
    def pretrain_config_path(self) -> Path:
        return self.config_dir / "pretrain.yaml"

    @computed_field
    @property
    def train_config_path(self) -> Path:
        return self.config_dir / "train.yaml"

    def get_sample_path(self, sample_name: str) -> Path:
        return self.experiment_samples_dir / f"{sample_name}.json"

    def get_attribute_dir(self, stage: str, mkdir=True) -> Path:
        path = self.attributes_dir / stage
        if mkdir:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def get_concept_model_dir(self, concept_name) -> Path:
        return self.model_gnn_dir / concept_name

    def get_prediction_path(
        self, dataset_name: str, model_name: str, sample_name: str
    ) -> Path:
        return self.prediction_dir / dataset_name / model_name / sample_name

    def get_result_path(
        self, dataset_name: str, model_name: str, sample_name: str
    ) -> Path:
        return self.result_dir / model_name / dataset_name / sample_name
