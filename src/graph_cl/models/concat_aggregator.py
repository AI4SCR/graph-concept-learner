import torch.nn as nn
from .mlp import MLP


class ConcatAggregator(nn.Module):
    """
    Aggregator that concatenates input tensors before passing them through an MLP.

    Attributes:
        mlp (nn.Module): MLP module.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Concatenates the graph embeddings and passes the output through an MLP.
        Returns the predicted unnomralized class probabilities for a batch of graph embeddings.
    """

    def __init__(
        self, emb_size, num_concepts, mlp_num_layers, num_classes, mlp_act, **kwargs
    ):
        """
        Initializes the ConcatAggregator module.

        Args:
            emb_size (int): The dimension of the input tensors.
            num_concepts (int): Number of concepts.
            num_layers (int): The number of layers in the MLP.
            num_classes (int): The dimension of the output tensor.
            mlp_act (nn.Module): The activation function to use in the MLP.
        """
        super().__init__()
        self.mlp = MLP(
            in_channels=emb_size * num_concepts,
            num_layers=mlp_num_layers,
            out_channels=num_classes,
            activation=mlp_act,
        )

    def forward(self, x):
        """
        Forward pass of the aggregator.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_concepts, emb_size).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_classes), the predicted unnomralized class probabilities for a batch of graph embeddings.
        """
        batch_size, num_concepts, emb_size = x.shape
        # Reshape to (batch_size, num_concepts * emb_size)
        concatenated = x.reshape(batch_size, -1)
        return self.mlp(concatenated)
