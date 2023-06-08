import torch
import torch.nn as nn
from .mlp import MLP


class LinearAggregator(nn.Module):
    """
    A PyTorch module that performs linear combination over the graph embedding for each of the concepts.
    Essentially this computes for each batch a tensor that is a linear combination of the graph embedding in each batch.

    Attributes:
        weights (nn.Parameter): A parameter tensor of shape (num_concepts,). It is learned during training.
        mlp (nn.Module): An MLP module that is used to transform the linearly combined graph embeddings.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Performs linear combination over the graph embeddings
        and passes the output through an MLP. Returns the predicted unnomralized class probabilities for a batch of graph embeddings.
    """

    def __init__(
        self,
        num_concepts: int,
        emb_size: int,
        mlp_num_layers: int,
        num_classes: int,
        mlp_act: nn.Module,
        **kwargs,
    ):
        """
        Initializes the LinearAggregator module.

        Args:
            num_concepts (int): The number of concepts in the input tensor.
            emb_size (int): The embedding size of the concepts.
            num_layers (int): The number of layers in the MLP module.
            num_classes (int): The number of output classes.
            mlp_act (nn.Module): The activation function to use in the MLP module.
        """
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_concepts))
        self.mlp = MLP(
            in_channels=emb_size,
            num_layers=mlp_num_layers,
            out_channels=num_classes,
            activation=mlp_act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs linear combination over the graph embedding for each of the concepts, and passes the
        output through an MLP.

        Args:
            x (torch.Tensor): The input tensor, of shape (batch_size, num_concepts, emb_size).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_classes), the predicted unnomralized class probabilities for a batch of graph embeddings.
        """
        x = torch.matmul(x.transpose(1, 2), self.weights)
        return self.mlp(x)
