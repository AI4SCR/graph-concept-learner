from pydantic import BaseModel
from .MixIns import FromYamlMixIn


class Graph(BaseModel):
    topology: str
    params: dict


class ConceptConfig(BaseModel, FromYamlMixIn):
    class Filter(BaseModel):
        col_name: str
        include_labels: list[str]

    name: str
    graph: Graph
    filter: Filter


# model = ConceptConfig()
# import yaml
# yaml_schema = yaml.dump(ConceptConfig.model_json_schema(), sort_keys=False)
#
# # Save the YAML schema to a file
# with open('/Users/adrianomartinelli/Downloads/schema.yaml', 'w') as f:
#     f.write(yaml_schema)
