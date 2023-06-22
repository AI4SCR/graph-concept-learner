import networkx as nx
import random
from torch_geometric.utils.convert import from_networkx

# Create an empty graph
graph = nx.Graph()

# Add nodes with attributes
# Generate a list of numbers from which to sample
numbers = list(range(1, 101))

# Sample 10 random integers without replacement
num_nodes = 10
random_integers = random.sample(numbers, num_nodes)

for node_id in random_integers:
    attributes = {
        "attribute1": random.randint(1, 100),
        "attribute2": random.uniform(0.0, 1.0),
        "attribute3": random.uniform(0.0, 1.0),
    }
    graph.add_node(node_id, **attributes)

# Add random edges
num_edges = 15  # Number of edges in the graph

for _ in range(num_edges):
    source = random.sample(random_integers, 1)[0]
    target = random.sample(random_integers, 1)[0]
    if source != target and not graph.has_edge(source, target):
        graph.add_edge(source, target)

# Print node attributes
for node_id, attributes in graph.nodes(data=True):
    print(f"Node {node_id}: {attributes}")

# Print edge list
print("Edges:")
for edge in graph.edges():
    print(edge)

g = from_networkx(graph, group_node_attrs=all)
