# %%
import torch
from torch import nn
from torch import Tensor
from einops import repeat
from .mlp import MLP


class PrependClassToken(nn.Module):
    """
    Preprocessing module for a transformer. Prepends class token to a batch of graph embeddings.

    Attributes:
        weights (nn.Parameter): A parameter tensor of shape (num_concepts,). It is learned during training.
        mlp (nn.Module): An MLP module that is used to transform the linearly combined graph embeddings.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: When the `forward` function is called a torch.Tensor with
        size (batch_size, num_concepts + 1, emb_size) is returned.
        Input to `forward` function should be a torch.Tensor of shape (batch_size, num_concepts, emb_size).
    """

    def __init__(self, emb_size: int):
        """
        Initializes the PrependClassToken module.

         Args:
            emb_size (int): size of the embedding size of each token.
        """
        super(PrependClassToken, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        """
        Prepends class token to a batch of graph embeddings.

        Args:
            x: input tensor of shape (batch_size, num_concepts, emb_size) and

        Returns:
            Tensor with size (batch_size, num_concepts + 1, emb_size)
        """

        # Get batch size
        b, _, _ = x.shape

        # Instantiate class tokens
        cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=b)

        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        return x


class ConceptGraphTransformer(nn.Module):
    """
    Integrates three `nn.Module` modules:
        - `PrependClassToken`: a class token prepender,
        - `TransformerEncoder`: a transformer encoder and
        - `ClassificationHead`: a classification head.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: `forward` should be called on a toch.Tensor of shape
        (batch_size, num_concepts, emb_size). Returns a torch.Tensor of shape
        (batch_size, n_classes), the predicted unnomralized class probabilities for a batch of graph embedding.
    """

    def __init__(
        self,
        emb_size: int,
        n_heads: int,
        depth: int,
        num_classes: int,
        scaler: int,
        mlp_num_layers: int,
        mlp_act: nn.Module,
        **kwargs,
    ):
        """
        Initializes a ConceptGraphTransformer module.

        Args:
            emb_size (int): Size of the token embeddings.
            n_heads (int): Number or MultiHeadAttention heads.
            depth (int): Number of staked TransformerEncoderLayer stacked.
            n_classes (int): Number of classes in the classification task.
            scaler (int): scaler * emb_size = the dimension of the feedforward network model.
        """
        super(ConceptGraphTransformer, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=scaler * emb_size,
        )

        self.transformer = nn.Sequential(
            PrependClassToken(emb_size),
            nn.TransformerEncoder(encoder_layer, num_layers=depth),
        )

        self.mlp = MLP(
            in_channels=emb_size,
            num_layers=mlp_num_layers,
            out_channels=num_classes,
            activation=mlp_act,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes predictions for a batch of graph embeddings.

        Args:
            x: input tensor of shape (batch_size, num_concepts, emb_size)

        Returns:
            Tensor of size (batch_size, n_classes), the predicted unnomralized class probabilities for a batch of graph embeddings.
        """

        x = self.transformer(x)
        # Slice output. Take only first CLSS token
        x = x[:, 0, :]
        return self.mlp(x)
