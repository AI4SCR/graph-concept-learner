import torch
import torch.nn as nn
from torch_geometric.data import Data
from graph_cl.models.beta_transformer import ConceptGraphTransformer
from graph_cl.models.linear_aggregator import LinearAggregator
from graph_cl.models.concat_aggregator import ConcatAggregator

aggregators = {
    "transformer": ConceptGraphTransformer,
    "linear": LinearAggregator,
    "concat": ConcatAggregator,
}

activation_functions = {"relu": nn.ReLU(), "tanh": nn.Tanh()}


class GraphConceptLearner(nn.Module):
    """
    Graph concept learner model which integrates concept wise GNN models to compute concept embeddings
    which are the aggregated to compute a prediction.

    Attributes:
        concept_learners (nn.ModuleDict): a dictionary of concept names and their corresponding GNN models.
        config (dict): a dictionary of configuration parameters for the aggregator model.

    Methods:
        forward(batch: torch.Tensor) -> torch.Tensor: computes the prediction of the model for a given batch.
    """

    def __init__(
        self, concept_learners: nn.ModuleDict, config: dict, device: torch.device
    ):
        """
        Initializes the GraphConceptLearner model with the given concept learners and aggregator.

        Args:
            concept_learners (nn.ModuleDict): a dictionary of concept names and their corresponding GNN models.
            config (dict): a dictionary of configuration parameters for the aggregator model.
            device (torch.device): device on which to load the empty graph embeddings
        """

        super().__init__()
        self.concept_learners = concept_learners

        # Get model class
        agg_class = aggregators[config["aggregator"]]

        # Add mlpo activation function to the config dictionary
        config["mlp_act"] = activation_functions[config["mlp_act_key"]]

        # Init model
        self.aggregator = agg_class(**config)

        # Get some aditionalinformation from config
        self.num_concepts = config["num_concepts"]
        self.emb_size = config["emb_size"]
        self.device = device

    def forward(self, batch):
        """
        Computes the prediction of the model for a given batch.

        Args:
            batch (graph_cl.datasets.concept_set_dataset.ConceptSetDatumBatch): a batch of graphs from all concepts in a ConceptSetDatumBatch object.

        Returns:
            torch.Tensor: a tensor of shape (batch_size, num_classes) containing the prediction for each input sample.
        """

        # For a batch of samples generate graph embedding for every concept
        embeddings = torch.empty(
            (len(batch.y), self.num_concepts, self.emb_size), device=self.device
        )

        for insert_at_dim, item in enumerate(self.concept_learners.items()):
            # Unpack concept name and corresponding model
            concept, model = item

            # Get concept sepcific data from Paradigm_DatumBatch
            x = batch[f"{concept}__x"]
            edge_index = batch[f"{concept}__edge_index"]
            x_batch = batch[f"{concept}__x_batch"]

            # Integrate data in to Data object
            data = Data(x=x, edge_index=edge_index, batch=x_batch)

            # Save resulting embedding of shape (batch_size, emb_size)
            batch_embeddings = model(data)

            # check that there is no nan in output
            assert not torch.isnan(
                batch_embeddings
            ).any(), f"NaN's in batch_embeddings. Printing batch graph embeddings for concept: {concept}. Batch embbedings {batch_embeddings}"

            # Insert the graph emebdding into tensor
            embeddings[:, insert_at_dim, :] = batch_embeddings

        # Pass embeddings through aggregator
        y_pred = self.aggregator(embeddings)
        # assert not torch.isnan(y_pred).any(), f"Something in y_pred is NaN. Printing y_pred: {y_pred}"
        return y_pred
