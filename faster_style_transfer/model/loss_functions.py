'''Define the loss functions for the feature extraction network.'''
import torch


def style_transfer_loss(inter_layers, feature_layers, style_layers,
                                                            style_weight=1.0):
    '''
    The total style transfer loss.

    Args:
        inter_layers: dict. Weights (the value: 4-D tensor) from the Perceptual
            Network's intermediate layers (the key: int).
        feature_layers: list of int. Layers from which content features are extracted.
        style_layers: list of int. Layers from which style features are extracted.
        style_weight: float. The weight of the style loss. Defaults to 1.0

    Returns:
        loss: float. The style transfer loss.
    '''
    # Get the first intermediate layer weights (4-D tensor).
    first_layer = next(iter(inter_layers.values()))
    # Get the number of content/output images' batch size.
    # -----
    # Each of "inter_layers" values contains the weights for some intermediate
    # layer. Those weights are computed for the content and its output (batch
    # of size "B") and one style image (stacked together respectively).
    # That is, we subtract 1 (the style image) and divide by 2 (content and
    # its output) to get the batch size.
    B = (first_layer.shape[0] - 1) // 2

    # Each "inter_layers" value is a 4-D tensor of shape (B+B+1, C, W, H). That
    # is, the 1st dimension is: 2*B [for content and output] + 1 [for style].
    # -----
    # We need to extract each image type's features individually.
    # For that we rearrange "inter_layers" in the way that each layer (the key)
    # contains a list of 3 elements (the value): Extracted features of the
    # content, the output and the style image respectively.
    inter_layers = {num_layer: torch.split(out_layer, [B, B, 1]) for \
                    num_layer, out_layer in inter_layers.items()}

    content = content_loss(inter_layers, feature_layers)
    style = style_loss(inter_layers, style_layers)

    return style_weight * style + content



def content_loss(inter_layers, feature_layers):
    '''
    Content loss between the content image and the output (synthesized) image.

    Args:
        inter_layers: dict. Weights (value: list of 3 tensors) from the Perceptual
            Network's intermediate layers (key: int).
        feature_layers: list of int. Layers from which content features are
            extracted.

    Returns:
        loss: float. The content loss.
    '''
    loss = 0

    # The total content loss will be the sum over all features' loss between
    # the content image and the output image.
    for layer_num in feature_layers:
        # Get the current "layer_num" extracted features.
        curr_content = inter_layers[layer_num][0]  # Content image's features.
        curr_output = inter_layers[layer_num][1]  # Output image's features.

        _, C, H, W = curr_content.shape
        U = C * H * W

        # Current feature loss is the (squared, normalized) Euclidean distance
        # between the content and the output features.
        loss += ((curr_output - curr_content) ** 2).sum() / U

    return loss



def style_loss(inter_layers, style_layers):
    '''
    Style loss between the style image and the output (synthesized) image.

    Args:
        inter_layers: dict. Weights (value: list of 3 tensors) from the Perceptual
            Network's intermediate layers (key: int).
        style_layers: list of int. Layers from which style features are extracted.

    Returns:
        loss: float. The style loss.
    '''
    loss = 0

    # The total style loss will be the sum over all features' loss between
    # the style image and the output image.
    for layer_num in style_layers:
        # Get the current "layer_num" extracted features (resp. style, output).
        # In the style loss, features' gram matrices are used instead of the
        # raw weights' tensors.
        curr_style = gram_matrix(inter_layers[layer_num][2])
        curr_output = gram_matrix(inter_layers[layer_num][1])

        # Current style loss is the squared Frobenius Norm between the style
        # and the output gram matrices.
        loss += ((curr_output - curr_style) ** 2).sum()

    return loss



def gram_matrix(x):
    '''
    Gram matrix of a tensor representing images. Used to compute the style loss.

    Args:
        x: 4-D tensor of shape (B, C, W, H). Represents the batch (B) of weights
            with channels (C), width (W) and height (H).

    Returns:
        gram: 2-D tensor of shape (C, C). The gram matrix.
    '''
    B, C, H, W = x.shape
    x = x.reshape(B, C, H*W)
    # Compute the normalized gram matrix, a 2-D tensor of shape (C, C).
    gram = (x @ torch.transpose(x, 1, 2)) / (C*H*W)
    return gram
