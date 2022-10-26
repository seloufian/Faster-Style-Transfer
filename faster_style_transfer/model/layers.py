'''Define used modules (layers) in the style transfer network.'''
from math import ceil

import torch
import torch.nn as nn


class CondInstanceNorm(nn.Module):
    '''
    Conditional Instance Normalization layer.
    '''
    def __init__(self, num_features, num_styles):
        '''
        Cond Instance Norm layer's constructor.

        Args:
            num_features: int. Number of input features to be normalized.
            num_styles: int. Number of styles handled by the Cond Instance Norm.
        '''
        super(CondInstanceNorm, self).__init__()

        self.num_styles = num_styles

        # The core component of the Cond Instance Norm is a list of Batch Norm layers.
        # That is, a separate Batch Norm is learned for each of the "num_styles".
        # Note that the Batch Norm list is wrapped in a "nn.ModuleList" so that
        # it can be tracked by PyTorch (in the backward pass).
        self.norm_layers = nn.ModuleList([nn.BatchNorm2d(num_features) for \
                                        _ in range(num_styles)])

    def forward(self, in_data):
        '''
        Perform a forward pass through the Cond Instance Norm layer.

        Args:
            in_data: dict. Input data for the forward pass. Contains:
                x: 4-D tensor of shape (B, num_features, W, H).
                num_style: Two possible types:
                    int. Style number, between 0 and "num_styles" (excluded).
                    list of int. Styles numbers, used for the convex combination.
                        This feature is defined only in the evaluation mode.
                weights: list of int, optional. Convex combination coefficients.
                    The weights must sum to 1, and be specified for all (or one less)
                    the "num_style" items. If the weight for the last style is not
                    specified, it is computed automatically.

        Returns:
            forward_output: 4-D tensor. Result of Cond Instance Norm's application
                (with selected "num_style") on the input "x" (in "in_data").

        Raises:
            KeyError: If the convex combination weights (coefficients) are not defined.
            TypeError: If the given "num_style" for the convex combination is not a list.
            ValueError:
                If the "num_style" value(s) is not between 0 and "num_styles" (excluded).
                If the convex combination feature is used in the training mode.
                If the weights' length does not equal (or is one less) the "num_style" length.
                If the weights' sum exceeds 1.
        '''
        x, num_style = in_data['x'], in_data['num_style']

        if isinstance(num_style, int):
            if not (0 <= num_style < self.num_styles):
                raise ValueError('The "style number" must be between '
                                f'0 and {self.num_styles} (excluded).')

            # Apply the Cond Instance Norm on "x" with selected "num_style".
            # That is, we use the Batch Norm layer which corresponds to "num_style".
            # In this way, the model will learn a separate input's scaling/shifting
            # for each style (of the total "num_styles").
            forward_output = self.norm_layers[num_style](x)

        else:  # The forward pass uses styles' convex combination.
            if not isinstance(num_style, list):
                raise TypeError("Styles' convex combination requires a 'list' of "
                               f"style numbers. Given '{type(num_style)}' instead.")

            if self.training:
                raise ValueError("Styles' convex combination is defined "
                                 "only in the evaluation mode.")

            if not all(0 <= style < self.num_styles for style in num_style):
                raise ValueError('All the "style number" values must be between '
                                f'0 and {self.num_styles} (excluded).')

            if 'weights' not in in_data:
                raise KeyError("Styles' convex combination requires a 'weights' list.")

            weights = in_data['weights']

            num_style_len = len(num_style)
            if len(weights) not in [num_style_len, num_style_len-1]:
                raise ValueError("The styles' convex combination weights "
                                f"list's length must be {num_style_len} or "
                                f'{num_style_len-1}. Given {len(weights)} instead.')

            if sum(weights) > 1:
                raise ValueError("The styles' convex combination "
                                 "weights sum must not exceed 1.")

            # When the weight for the last style isn't specified, it has to be computed.
            if len(weights) == num_style_len-1:
                # In the convex combination, the coefficients sum to 1.
                # Given all the coefficients except one, it can be easily computed.
                last_style_weight = 1 - sum(weights)
                # Add the last style's weight to the weights list.
                weights.append(last_style_weight)

            # Define a container, which will store temporarily the styles'
            # convex combination parts (each "weight * style" separately).
            forward_output = []
            # Compute the convex combination parts.
            # That is, each style's conditional instance layer's output
            # is multiplied by its weight (coefficient value).
            for idx, style in enumerate(num_style):
                forward_output.append(weights[idx] * self.norm_layers[style](x))

            # Compute the styles' convex combination, the sum of all parts.
            forward_output = torch.stack(forward_output).sum(dim=0)

        return forward_output



class SamePadConv2d(nn.Module):
    '''
    Same reflection padded 2-D convolution.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 activation=nn.ReLU(), num_styles=0):
        '''
        Same reflection padded 2-D conv's constructor.

        Args:
            in_channels: int. Number of input channels to the 2-D conv.
            out_channels: int. Number of output channels of the 2-D conv.
            kernel_size: int. Kernel size of the 2-D conv, must be an odd number.
            stride: int. Stride value of the 2-D conv.
            activation: "torch.nn" method. Activation to use on the 2-D conv's output.
                Common values: `nn.ReLU()` (default), `nn.Sigmoid()`,
                               `nn.Identity()` (do not apply an activation), etc.
            num_styles: int. Styles number of the Cond Instance Norm.
                Assigning "0" (or a negative number) means not using the normalization.
        '''
        super(SamePadConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride

        # In the same-padded convolution, the "padding" is determined and done
        # manually (on the input) in the forward pass.
        # That is, we choose to not apply padding in the standard "nn.Conv2d".
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=0)
        self.activation = activation

        self.use_norm = True if num_styles > 0 else False
        if self.use_norm:
            self.normalization = CondInstanceNorm(out_channels, num_styles)

    def forward(self, in_data):
        '''
        Perform a forward pass through the same-padded 2-D convolution.

        Args:
            in_data: dict. Input data for the forward pass. Contains two elements:
                x: 4-D tensor of shape (B, num_features, W, H).
                num_style: int. Style number (for the Cond Instance Norm),
                           between 0 and "num_styles" (excluded).

        Returns:
            dict. Two elements: Output "x" of the forward pass and "num_style" (unchanged).
        '''
        x = in_data['x']

        # Apply the "same_padding" on the input "x". So, once passed through
        # "conv2d", the output will have spatial dimensions which corresponds
        # to the same-padded convolution.
        x = same_padding(x, kernel=self.kernel_size, stride=self.stride)
        x = self.conv2d(x)

        in_data['x'] = self.activation(x)

        if self.use_norm:
            in_data['x'] = self.normalization(in_data)

        return in_data



class ResidualBlock(nn.Module):
    '''
    Style transfer network's residual block.

    The residual block is composed of two layers:
        Same reflect pad 2-D conv, with ReLU activation and without normalization.
        Same reflect pad 2-D conv, without activation and without normalization.
    '''
    def __init__(self, insize=128, outsize=128, kernel=3):
        super(ResidualBlock, self).__init__()

        self.conv1 = SamePadConv2d(insize, outsize, kernel_size=kernel,
                                   stride=1, num_styles=0)
        self.conv2 = SamePadConv2d(outsize, outsize, kernel_size=kernel,
                           stride=1, activation=nn.Identity(), num_styles=0)

    def forward(self, in_data):
        conv1_out_data = self.conv1(in_data)
        conv2_out_data = self.conv2(conv1_out_data)

        # Add the input and the output (as in the general residual blocks logic).
        conv2_out_data['x'] += in_data['x']

        return conv2_out_data



class Upsampling(nn.Module):
    '''
    Nearest-neighbor interpolation upsampling used in the style transfer network.
    '''
    def __init__(self, factor=2):
        super(Upsampling, self).__init__()

        self.upsample = nn.UpsamplingNearest2d(scale_factor=factor)

    def forward(self, in_data):
        in_data['x'] = self.upsample(in_data['x'])
        return in_data



def same_padding(in_images, kernel=3, stride=1):
    '''
    Reflection same padding.

    Note: This function operates on square (1:1 ratio) images (width = height).

    Args:
        in_images: 4-D tensor of shape (B, C, W, H). Input images to be padded.
        kernel: int. Kernel size of the 2-D conv which will be applied on the
                padded "in_images" (Used to determine the padding).
        stride: int. Stride of the 2-D conv (same usage as for the "kernel").

    Returns:
        Padded "in_images", a 4-D tensor of shape (B, C, W+p, H+p), where "p" is
        the determined padding.
    '''
    # Get the width/height of the input (square) images.
    D = in_images.shape[-1]

    # Compute the output width/height (formula used in the same-padding).
    Dout = ceil(D / stride)

    # Given "Dout" (output H/W), "D" (input H/W), "kernel" and "stride", we can
    # determine the "padding" (the only unknown) from the 2-D convolution formula:
    # Dout = ceil([D - kernel + 2 * padding] / 2) + 1
    padding = ceil(((Dout - 1) * stride - D + kernel) / 2)

    # Define the reflection padding.
    pad = nn.ReflectionPad2d(padding)

    # Apply the defined reflection padding on input images.
    return pad(in_images)
