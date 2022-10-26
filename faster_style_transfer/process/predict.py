'''Define the style transfer network prediction/evaluation function.'''
import torch
import torchvision.transforms as T
from torchvision import io


from .utils import center_crop


def predict(
    model,
    content_images_path,
    style_index,
    weights=None,
    convert=True
):
    '''
    Apply the style transfer.

    Args:
        model: dict. Style transfer network built model, as returned by
            `build_model.build_model` function.
        content_images_path: list of str. Content images paths (on which the style
            transfer will be applied).
        style_index: Style(s) which will be applied on the content images.
            Two possible types:
                int. Style number (default usage, one style application).
                list of int. Styles numbers, used for the convex combination
                    (multiple -weighted- styles application).
        weights: list of int, optional. Convex combination coefficients.
            The weights must sum to 1, and be specified for all (or one less)
            the "style_index" items. If the weight for the last style is not
            specified, it will be computed automatically.
        convert: bool. Whether to convert the output to `PIL` images or not.

    Returns:
        stylized_images: list (with the same length as `content_images_path`) of:
            `PIL` images, if `convert` is `True`.
            Tensors with shape (3, 256, 256), if `convert` is `False`.
    '''

    # Get the style transfer network and the device it is loaded on.
    style_transfer = model['style_transfer']
    device = model['device']

    # Store the processed (read and transformed) content images.
    images = []

    for image_path in content_images_path:
        # Open the current image as a PyTorch tensor.
        image = io.read_image(image_path)
        # Center crop the image. The result is a tensor of shape (3, 256, 256)
        image = center_crop(image)

        images.append(image)

    # Stack the processed images list
    # The result is a tensor of shape (<len(images)>, 3, 256, 256)
    images = torch.stack(images).to(device=device)

    # Apply the chosen style (specified by its index) on the content images.
    with torch.no_grad():
        in_data = {'x': images, 'num_style': style_index}
        # If the convex combination style transfer was chosen, add the weights
        # to the style transfer model's input data.
        if weights is not None:
            in_data['weights'] = weights
        # Style transfer network's output has a similar shape to the "images" tensor.
        stylized_images = style_transfer.style_transfer_net(in_data)

    # Split the grouped content images tensor.
    # The result is a list with <number_content_images> elements, each is
    # a tensor of shape (3, 256, 256)
    stylized_images = stylized_images.split(1)
    stylized_images = [image.squeeze(0) for image in stylized_images]

    if convert:  # If the output should be converted to `PIL` images.
        # Convert the result list's tensors to `PIL` images.
        converter = T.ToPILImage()
        stylized_images = [converter(image) for image in stylized_images]

    return stylized_images
