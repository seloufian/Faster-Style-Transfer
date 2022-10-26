'''Define the style transfer module.'''
import torch
import torch.nn as nn

from .perceptual_network import PerceptualNetwork
from .network import StyleTransferNetwork


class StyleTransferModule(nn.Module):
    '''
    The style transfer module.
    Architecture defined in the paper <https://arxiv.org/abs/1610.07629>.

    Composed of two components:
        - Perceptual network: For the feature extraction from images.
        - Style transfer network: Apply an artistic style on images.
    '''
    def __init__(self, inter_layers_num, num_styles):
        '''
        Style transfer module's constructor.

        Args:
            inter_layers_num: list of int. Specifies from which perceptual
                network's layers the features will be extracted.
                The values must be between 0 and 23 (both included).
            num_styles: int. Number of styles handled by the style transfer
                network. Greater or equal to "1".
        '''
        super(StyleTransferModule, self).__init__()

        self.perceptual_net = PerceptualNetwork(inter_layers_num)
        self.style_transfer_net = StyleTransferNetwork(num_styles)
        self.style_transfer_net.apply(init_isotropic_gaussian)

    def forward(self, x, num_style, s):
        '''
        Perform a forward pass through the style transfer module.

        Args:
            x: 4-D tensor of shape (B, 3, 256, 256). Input batch of content images.
            num_style: int. Style number which corresponds to the style image "s".
            s: 4-D tensor of shape (1, 3, 256, 256). Input style image.

        Returns:
            inter_layers: dict (key: int, value: 4-D tensor). Output of the
                perceptual network. Weights for each "inter_layers_num".
        '''
        # Perform a forward pass through the style transfer network.
        # The output (y_hat) is the stylized (synthesized) images.
        in_data = {'x': x, 'num_style': num_style}
        y_hat = self.style_transfer_net(in_data)

        # stack -respectively- vertically [the batch of content/output images](2*B)
        # and [the style image](1).
        # "in_x" is a 4-D tensor of shape (2*B+1, 3, 256, 256)
        in_x = torch.vstack((x, y_hat, s))

        # Perform a forward pass through the perceptual network.
        inter_layers = self.perceptual_net(in_x)
        return inter_layers


def init_isotropic_gaussian(layer):
    '''
    Isotropic Gaussian weight initialization.
    Normal distribution with `mu = 0` and `sigma = 0.01`.
    '''
    # The initialization is applied only on 2-D conv layers.
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.01)
