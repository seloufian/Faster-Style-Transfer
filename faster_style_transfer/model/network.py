'''Define the style transfer network.'''
import torch.nn as nn

from .layers import SamePadConv2d, ResidualBlock, Upsampling


class StyleTransferNetwork(nn.Module):
    '''
    Style transfer network.
    Applies chosen style on a batch of input images.
    '''
    def __init__(self, num_styles):
        '''
        Style transfer network's constructor.

        Args:
            num_style: int. Number of styles of the network, greater or equal to "1".
        '''
        super(StyleTransferNetwork, self).__init__()

        self.contract = nn.Sequential(
            SamePadConv2d(3, 32, kernel_size=9, stride=1, num_styles=num_styles),
            SamePadConv2d(32, 64, kernel_size=3, stride=2, num_styles=num_styles),
            SamePadConv2d(64, 128, kernel_size=3, stride=2, num_styles=num_styles)
        )

        self.residual = nn.Sequential(
            *[ResidualBlock() for _ in range(5)]
        )

        self.expand = nn.Sequential(
            Upsampling(),
            SamePadConv2d(128, 64, kernel_size=3, stride=1, num_styles=num_styles),
            Upsampling(),
            SamePadConv2d(64, 32, kernel_size=3, stride=1, num_styles=num_styles)
        )

        self.output = SamePadConv2d(32, 3, kernel_size=9, stride=1,
                            activation=nn.Sigmoid(), num_styles=0)

    def forward(self, in_data):
        '''
        Perform a forward pass through the style transfer network.

        Args:
            in_data: dict. Input data for the forward pass. Contains two elements:
                x: 4-D tensor of shape (B, 3, W, H). Input images.
                num_style: Two possible types:
                    int. Style number, between 0 and "num_styles" (excluded).
                    list of int. Styles numbers, used for the convex combination.
                        This feature is defined only in the evaluation mode.
                weights: list of int, optional. Convex combination coefficients.
                    The weights must sum to 1, and be specified for all (or one less)
                    the "num_style" items. If the weight for the last style is not
                    specified, it is computed automatically.

        Returns:
            A 4-D tensor of shape (B, 3, W, H). Artistic style applied to the
            input images.
        '''
        in_data = self.contract(in_data)
        in_data = self.residual(in_data)
        in_data = self.expand(in_data)
        out_data = self.output(in_data)
        return out_data['x']
