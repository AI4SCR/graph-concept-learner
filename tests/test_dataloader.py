from graph_cl.dataloader.ConceptDataModule import ConceptDataModule
from pathlib import Path

data_dir = Path("/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/")
target = "ERStatus"

dm = ConceptDataModule(data_dir=data_dir, target=target)
dm.setup(stage="fit")
