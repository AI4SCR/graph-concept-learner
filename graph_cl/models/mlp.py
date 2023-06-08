import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron (MLP) with an arbitrary number of linear layers.

    Args:
        in_channels (int): The number of input features.
        num_layers (list): Number of layers in the MLP.
        out_channels (int): The number of output features.
        activation (nn.Module): The activation function to use.

    Attributes:
        layers (nn.Sequential): The sequential module containing the MLP layers.

    """

    def __init__(self, in_channels, num_layers, out_channels, activation=nn.ReLU()):
        super(MLP, self).__init__()

        # List of layers and activations
        layers = []

        # Append all layers exept the last one
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_channels, in_channels))
            layers.append(activation)

        # Append last layer
        layers.append(nn.Linear(in_channels, out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Passes the input tensor through the MLP layers.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """

        return self.layers(x)
