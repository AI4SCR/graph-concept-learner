from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, computed_field
from dotenv import find_dotenv


class ProjectPathFactory(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(".env", usecwd=True),
        protected_namespaces=("settings_",),
        extra="ignore",
    )

    data_dir: Path | str

    def init(self):
        dirs = self.get_dirs()
        for d in dirs:
            path = getattr(self, d)
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_dirs(cls):
        return [i for i in cls.__dict__["model_computed_fields"] if i.endswith("_dir")]

    @computed_field
    @property
    def datasets_dir(self) -> None | Path:
        return self.data_dir / "datasets"

    @computed_field
    @property
    def concepts_dir(self) -> Path:
        return self.data_dir / "concepts"

    @computed_field
    @property
    def experiments_dir(self) -> Path:
        return self.data_dir / "experiments"

    def get_concept_config_path(self, concept_name) -> None | Path:
        if self.concepts_dir is None:
            return None
        return self.concepts_dir / f"{concept_name}.yaml"

    @field_validator("data_dir")
    @classmethod
    def to_path(cls, v: str | Path) -> Path:
        return Path(v)
