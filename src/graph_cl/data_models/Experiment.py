from pathlib import Path

from ai4bmr_core.data_models.Experiment import Experiment
from pydantic import computed_field


class GCLExperiment(Experiment):
    @computed_field
    @property
    def split_info_path(self) -> Path:
        return self.experiment_dir / "split_info.parquet"

    @computed_field
    @property
    def attributes_dir(self) -> Path:
        return self.experiment_dir / "010_attributes"

    @computed_field
    @property
    def samples_dir(self) -> Path:
        return self.experiment_dir / "011_samples"

    @computed_field
    @property
    def data_config_path(self) -> Path:
        return self.configs_dir / "data.yaml"

    @computed_field
    @property
    def model_gnn_config_path(self) -> Path:
        return self.configs_dir / "model_gnn.yaml"

    @computed_field
    @property
    def model_gcl_config_path(self) -> Path:
        return self.configs_dir / "model_gcl.yaml"

    @computed_field
    @property
    def pretrain_config_path(self) -> Path:
        return self.configs_dir / "pretrain.yaml"

    @computed_field
    @property
    def train_config_path(self) -> Path:
        return self.configs_dir / "train.yaml"

    def get_attribute_dir(self, stage: str, mkdir=True) -> Path:
        path = self.attributes_dir / stage
        if mkdir:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def get_concept_model_dir(self, model_name: str, concept_name: str) -> Path:
        return self.get_model_dir(model_name) / concept_name
