import torch

core = "BaselTMA_SP43_149_X10Y6_223.pt"

c = f"/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/03_concept_graphs/contact_tumor_immune/{core}"
k = f"/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/03_concept_graphs/knn_all/{core}"
r = f"/Users/adrianomartinelli/data/ai4src/graph-concept-learner/data/03_concept_graphs/radius_tumor_immune/{core}"

cg = torch.load(c)
kg = torch.load(k)
rg = torch.load(r)

print(cg.num_nodes)
print(kg.num_nodes)
print(rg.num_nodes)
