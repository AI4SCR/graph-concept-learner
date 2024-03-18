from pathlib import Path

from ai4bmr_core.data_models.Project import Project
from pydantic import computed_field


class GCLProject(Project):
    @computed_field
    @property
    def concepts_dir(self) -> Path:
        return self.base_dir / "03_concepts"

    def get_concept_config_path(self, concept_name) -> Path:
        return self.concepts_dir / f"{concept_name}.yaml"


# note: we use this instance across the project.
project = GCLProject()
