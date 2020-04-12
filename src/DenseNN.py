import torch
import torch.nn as nn


class DenseNN(nn.Module):
    """
    A two layer dense neural network architecture template for general use
    """
    # TODO: add n_layers and make the constructor auto create the correct num layers
    def __init__(self, n_inputs, n_hidden, wide_connection=True):
        """
        Creates a new instance of the DenseNN class


        """
        nn.Module.__init__(self)
        self.wide_connection = wide_connection
        self.layer1 = nn.Linear(n_inputs, n_hidden)
        if self.wide_connection:
            self.layer2 = nn.Linear(n_inputs+n_hidden, n_hidden)
        else:
            self.layer2 = nn.Linear(n_hidden, n_hidden)

    def forward(self, inputs):
        """
        Takes an inputs batch and outputs runs them through the model
        """
        x = self.layer1(inputs)
        if self.wide_connection:
            x = self.layer2(torch.cat(tensors=(inputs, x), dim=1))
        else:
            x = self.layer2(x)
        return x
