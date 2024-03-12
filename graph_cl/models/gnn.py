import torch
import torch.nn as nn
from .mlp import MLP
from torch_geometric.utils import degree
from torch_geometric.nn import (
    GCN,
    GraphSAGE,
    GAT,
    GIN,
    PNA,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)

# Define dictionary with possible models.
models = {"GCN": GCN, "SAGE": GraphSAGE, "GAT": GAT, "GIN": GIN, "PNA": PNA}

# Define dictionary with possible pooling strategies.
pool_layers = {
    "global_add_pool": global_add_pool,
    "global_mean_pool": global_mean_pool,
    "global_max_pool": global_max_pool,
}


class GNN(nn.Module):
    """
    A graph neural network (GNN) without a classification head.

    Attributes:
        gnn (torch_geometric.nn.Module): The graph neural network model.
        pool (torch_geometric.nn.pool): The pooling layer used to pool node embeddings into a graph-level embedding.

    Methods:
        forward(x: torch.Tensor, edge_index, batch) -> torch.Tensor: Performs a forward pass through the GNN model.
    """

    def __init__(self, cfg):
        """
        Initializes the GNN model.

        Args:
            cfg (dict): A dictionary containing configuration parameters for the model.
            train_dataset: (graph_cl.datasets.ConceptDataset) The training dataset. Used to initialize the model in case the
            model is a PNA GNN.
        """
        # Super init
        super(GNN, self).__init__()

        # Get model class
        gnn_class = models[cfg["gnn"]]

        # If PNA then get adddtional parametes.
        if cfg["gnn"] == "PNA":
            raise NotImplementedError()
            get_additional_PNA_params(cfg, train_dataset)

        # Init GNN model
        self.gnn = gnn_class(**cfg)

        # Define pooling strategy
        self.pool = pool_layers[cfg["pool"]]

    def forward(self, data):
        """
        Performs a forward pass through the model.

        Args:
            torch_geometric.DataBatch object with attributes:
                x (torch.Tensor): The input node features.
                edge_index (torch.Tensor): The graph edge indices.
                batch (torch.Tensor): The node indices that correspond to each graph.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_classes). The predicted unnomralized class probabilities for a batch of graphs.
        """

        # 1. Obtain node embeddings
        x = self.gnn(data.x, data.edge_index)

        # 2. Pool node embedding into a graph level embedding
        x = self.pool(x, data.batch)

        return x


def get_additional_PNA_params(cfg, train_dataset):
    # TODO: Implement this function outside of the model class, or as a static method of the class
    raise NotImplementedError()
    # Specify aggregators and scalars
    cfg["aggregators"] = ["min", "max", "mean", "std"]
    cfg["scalers"] = ["identity", "amplification", "attenuation"]

    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    cfg["deg"] = deg


class GNN_plus_MPL(nn.Module):
    """
    A graph neural network (GNN) with a classification head.

    Attributes:
        GNN (graph_cl.models.gnn GNN): The graph neural network model.
        MLP (graph_cl.models.mlp MLP): The prediction head.

    Methods:
        forward(x: torch.Tensor, edge_index, batch) -> torch.Tensor: Performs a forward pass through the GNN model and the prediction head.
    """

    def __init__(self, cfg):
        """
        Initializes the GNN model and the prediction head.

        Args:
            cfg (dict): A dictionary containing configuration parameters for the model.
            train_dataset: (graph_cl.datasets.ConceptDataset) The training dataset. Used to initialize the model in case the
            model is a PNA GNN.
        """
        # Super init
        super().__init__()

        cfg["hidden_channels"] = cfg["in_channels"] * cfg["scaler"]

        self.gnn = GNN(cfg)
        self.mlp = MLP(
            in_channels=cfg["hidden_channels"],
            num_layers=cfg["num_layers_MLP"],
            out_channels=cfg["num_classes"],
        )

    def forward(self, data):
        """
        Performs a forward pass through the model.

        Args:
            torch_geometric.DataBatch object with attributes:
                x (torch.Tensor): The input node features.
                edge_index (torch.Tensor): The graph edge indices.
                batch (torch.Tensor): The node indices that correspond to each graph.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_classes). The predicted unnomralized class probabilities for a batch of graphs.
        """
        # Obtain graph embeddings for a batch of graphs
        x = self.gnn(data)

        # Obtain predicitons
        return self.mlp(x)
