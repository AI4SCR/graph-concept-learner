import torch
import torch.nn as nn
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

    def __init__(self, concept_learners: nn.ModuleDict, config: dict):
        """
        Initializes the GraphConceptLearner model with the given concept learners and aggregator.

        Args:
            concept_learners (nn.ModuleDict): a dictionary of concept names and their corresponding GNN models.
            config (dict): a dictionary of configuration parameters for the aggregator model.
        """

        super().__init__()
        self.concept_learners = concept_learners
        self.concept_names = list(self.concept_learners.keys())
        self.num_concepts = len(self.concept_names)

        # check and infer emb_size from first concept learner
        n_out = set(model.gnn.out_channels for model in self.concept_learners.values())
        assert len(n_out) == 1
        self.emb_size = n_out.pop()

        # Get model class
        agg_class = aggregators[config["aggregator"]]

        # Add mlp activation function to the config dictionary
        # getattr(nn, config["mlp_act_key"]) --> getattr(nn, "Tanh")
        config["mlp_act"] = activation_functions[config["mlp_act_key"]]

        # Init model
        self.aggregator = agg_class(emb_size=self.emb_size, **config)

    def forward(self, batch):
        """
        Computes the prediction of the model for a given batch.

        Args:
            batch (graph_cl.datasets.concept_set_dataset.ConceptSetDatumBatch): a batch of graphs from all concepts in a ConceptSetDatumBatch object.

        Returns:
            torch.Tensor: a tensor of shape (batch_size, num_classes) containing the prediction for each input sample.
        """

        # For a batch of samples generate graph embedding for every concept

        cont = []
        for concept, val in batch.items():
            cont.append(val.y)
            assert len(set(val.concept)) == 1

        n_samples = torch.stack(cont, dim=1).size(0)
        assert torch.stack(cont, dim=1).to("cpu").unique(dim=1).size(dim=1) == 1

        concept = self.concept_names[0]
        device = batch[concept].x.device.type
        embeddings = torch.empty(
            (n_samples, self.num_concepts, self.emb_size), device=device
        )

        for insert_at_dim, item in enumerate(self.concept_learners.items()):
            # Unpack concept name and corresponding model
            concept, model = item
            data = batch[concept]

            # Get concept specific data from Paradigm_DatumBatch
            # x = batch[f"{concept}__x"]
            # edge_index = batch[f"{concept}__edge_index"]
            # x_batch = batch[f"{concept}__x_batch"]

            # Integrate data in to Data object
            # data = Data(x=x, edge_index=edge_index, batch=x_batch)

            # Save resulting embedding of shape (batch_size, emb_size)
            batch_embeddings = model(data)

            # check that there is no nan in output
            assert not torch.isnan(
                batch_embeddings
            ).any(), f"NaN's in batch_embeddings. \nPrinting batch graph embeddings for concept: \n{concept}. \n\nBatch embeddings: \n{batch_embeddings}"

            # Insert the graph embedding into tensor
            embeddings[:, insert_at_dim, :] = batch_embeddings

        # Pass embeddings through aggregator
        y_pred = self.aggregator(embeddings)
        # assert not torch.isnan(y_pred).any(), f"Something in y_pred is NaN. Printing y_pred: {y_pred}"
        return y_pred
