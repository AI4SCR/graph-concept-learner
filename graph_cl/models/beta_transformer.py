import torch
from torch import nn
from torch import Tensor
from einops import repeat
from graph_cl.models.mlp import MLP
from typing import Optional, Callable
from torch.nn import functional as F


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


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    This class is almost an exact replica of nn.TransformerEncoderLayer.
    Only the forward and _sa_block are (minimaly) overwritten to allow for
    weight return. Probably a dangerous feature to add, use with care.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        need_weights: bool = False,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str or Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 0.00001,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device,
            dtype,
        )
        self.need_weights = need_weights

    # def __init__(self, need_weights: bool = False):
    #     super(nn.TransformerEncoderLayer, self).__init__()
    #

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).

        Shape:
            see the docs in Transformer class.
        """

        x = src
        if self.norm_first:
            attn_output, attn_output_weights = self._sa_block(self.norm1(x))
            x = x + attn_output
            x = x + self._ff_block(self.norm2(x))
        else:
            attn_output, attn_output_weights = self._sa_block(x)
            x = self.norm1(x + attn_output)
            x = self.norm2(x + self._ff_block(x))

        if self.need_weights:
            return x, attn_output_weights
        else:
            return x

    def _sa_block(
        self,
        x: Tensor,
    ) -> Tensor:
        # Call multi_head_attention_forward
        x, attn_output_weights = self.self_attn(
            x, x, x, need_weights=self.need_weights, average_attn_weights=False
        )

        # Return weights if self.need_weights == True
        if self.need_weights:
            return self.dropout1(x), attn_output_weights
        else:
            return (
                self.dropout1(x),
                attn_output_weights,
            )  # Here attn_output_weights will be None


class ConceptGraphTransformer(nn.Module):
    """
    Integrates three `nn.Module` modules:
        - `PrependClassToken`: a class token prepender,
        - `TransformerEncoder`: a transformer encoder and
        - `ClassificationHead`: a classification head.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: `forward` should be called on a torch.Tensor of shape
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
        need_weights: bool = False,
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
            need_weights (bool): flag for output attention filters from las layer
        """

        # Init and save need_weights to attributes
        super(ConceptGraphTransformer, self).__init__()
        self.need_weights = need_weights
        self.depth = depth

        # Initialize a single layer.
        # By default need_weights inside CustomTransformerEncoderLayer = False
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=emb_size,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=scaler * emb_size,
        )

        # Init the token prepender + encoder
        self.transformer = nn.Sequential(
            PrependClassToken(emb_size),
            nn.TransformerEncoder(encoder_layer, num_layers=depth),
        )

        # If need weights then set the last layer to true
        if need_weights:
            self.transformer[1].layers[depth - 1].need_weights = True

        # Init the MLP
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
            If self.need_weights == True then it also returns a tensor of attention weights of dimension (batch_size, n_heads, num_concepts + 1, num_concepts + 1)
        """

        # Slice output. Take only first CLSS token. Return weights if needed
        if self.need_weights:
            x, attn_output_weights = self.transformer(x)
            return self.mlp(x[:, 0, :]), attn_output_weights
        else:
            x = self.transformer(x)
            return self.mlp(x[:, 0, :])

    def return_attention_map(self, set_to: bool = True):
        """
        When this function is called with set_to=True the last layer of the transformer returns attention maps with the prediction.
        """
        self.transformer[1].layers[self.depth - 1].need_weights = set_to
        self.need_weights = set_to
