'''Define the feature extraction network.'''
from copy import deepcopy

import torch.nn as nn
from torchvision import models


class PerceptualNetwork(nn.Module):
    '''
    The feature extraction network. Takes a batch of input images and
    returns extracted features from the specified VGG-16 layers.
    '''
    def __init__(self, inter_layers_num):
        '''
        Perceptual network's constructor.

        Args:
            inter_layers_num: list of int. Specifies from which VGG-16's layers
                the features will be extracted. The values must be between 0
                and 23 (both included).

        Raises:
            ValueError: If `inter_layers_num` values are not between 0 and 23.
        '''
        super(PerceptualNetwork, self).__init__()

        if not all(0 <= layer_num <= 23 for layer_num in inter_layers_num):
            raise ValueError("Intermediate VGG-16 layers' numbers must be "
                             'between 0 and 23 (both included).')

        # Get the VGG-16 network, pre-trained on ImageNet.
        vgg16 = models.vgg16(weights='DEFAULT')
        # For memory usage and inference speed reasons, get only the necessary
        # VGG-16 layers. That is, only layers from the input to avg_pool_4 block
        # are retrieved (i.e. we get rid of avg_pool_5 and fully-connected
        # blocks).
        self.model = deepcopy(vgg16.features[:24])
        # The VGG-16 network is used as a feature extractor, so we won't its
        # weights being updated. That is, we freeze all the parameters.
        for param in self.model.parameters():
            param.requires_grad = False
        # Set VGG-16 to the evaluation mode.
        self.model.eval()

        # Register forward hooks for the specified intermediate layers.
        # This will save the specified layers' weights during the forward pass.
        for lnum in inter_layers_num:
            self.model[lnum].register_forward_hook(self._get_activation(lnum))
        # A dictionary which will contain for each specified intermediate layer
        # (the key), a 4-D tensor with its weights (the value).
        self.inter_layers = {}

    def _get_activation(self, num_layer):
        '''
        Save the specified "num_layer" (int) weights (4-D tensor) in the
        "inter_layers" dictionary.
        '''
        def hook(model, input, output):
            self.inter_layers[num_layer] = output
        return hook

    def forward(self, x):
        '''
        Perform a forward pass through the "Perceptual Network".

        Args:
            x: 4-D tensor of shape (B, 3, 256, 256). Represents the batch (B)
                of input images with RGB channels and width/height of 256.

        Returns:
            inter_layers: dict (key: int, value: 4-D tensor). Weights for each
                of "inter_layers_num".
        '''
        self.model(x)
        return self.inter_layers
