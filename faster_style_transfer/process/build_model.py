'''Define the style transfer network build function.'''
from .utils import dataset_loader, style_transfer_model, load_checkpoint


def build_model(
    style_path,
    mode='eval',
    content_path=None,
    transform=True,
    batch_size=16,
    shuffle=True,
    num_workers=1,
    checkpoint_path=None,
    feature_layers=[14],
    style_layers=[2, 7, 14, 21],
    style_weight=17,
    device='cuda',
):
    '''
    Build the style transfer model.
    For both 'train' and 'evaluation' (inference) modes.

    Args:
        style_path: str. Style images directory path.
            Accepted image extensions are: 'png' and 'jpg'.
        mode: str. Model's mode, either `train` or `eval` (evaluation).
        content_path: str. Content images directory path, depends on the `mode`:
            In the `train` mode, the `content_path` must be specified.
            In the `eval` mode, the `content_path` must be None (not specified).
        transform: bool. Whether to transform (center crop) the images or not.
        batch_size: int. Batch size, preferably a power of 2.
            In the `eval` mode, the `batch_size` parameter does not apply.
        shuffle: bool. Whether to shuffle the content images or not.
            In the `eval` mode, the `shuffle` parameter does not apply.
        num_workers: int. Number of parallel workers for content images loading.
            The recommended value is the number of CPU cores.
            In the `eval` mode, the `num_workers` parameter does not apply.
        checkpoint_path: str, optional. Checkpoint load path, depends on the `mode`:
            In the `train` mode, the `checkpoint_path` is optional.
                Defaults to None (train new model)
            In the `eval` mode, the `checkpoint_path` must be specified.
        feature_layers: list of int. VGG-16 layers for the content features extraction.
        style_layers: list of int. VGG-16 layers for the style features extraction.
        style_weight: float. Style loss weight. Defaults to 17
        device: str. Device which the network will be loaded on.

    Returns:
        model: dict. Built style transfer model, with keys:
            content_loader: StyleTransferDataset. Content images dataset.
            style_loader: StyleTransferDataset. Style images dataset.
            style_transfer: StyleTransferModule. Style transfer model.
            optimizer: Adam. Model's Adam optimizer.
            feature_layers: list of int. VGG-16 layers for the content features extraction.
            style_layers: list of int. VGG-16 layers for the style features extraction.
            style_weight: float. Style loss weight.
            iteration: int. Model iteration index value.
                Defaults to 0, if no `checkpoint_path` specified.
            iter_loss: dict. Training losses/time for each iteration.
                Defaults to [], if no `checkpoint_path` specified.
            device: str. Device on which the model and its optimizer are moved.

    Raises:
        ValueError:
            If the `mode` is not either 'train' or 'eval'.
            If the model is in `train` mode, and the `content_path` is not specified.
            If the model is in `eval` mode, and the `content_path` is specified
                and/or the `checkpoint_path` is not specified.
    '''

    if mode not in ['train', 'eval']:
        raise ValueError("The 'mode' must be either 'train' or 'eval'. "
                        f"Given '{mode}' instead.")

    if mode == 'train' and content_path is None:
        raise ValueError("The model is in 'train' mode, and the 'content_path' "
                        f"'is not specified ({content_path}).")

    if mode == 'eval':
        if content_path is not None:
            raise ValueError("The model is in 'eval' mode, and the "
                            f"'content_path' is specified ({content_path}).")
        if checkpoint_path is None:
            raise ValueError("The model is in 'eval' mode, and the "
                    f"'checkpoint_path' is not specified ({checkpoint_path}).")

    # Create the style/content image dataset loaders.
    image_dataset_loaders = dataset_loader(style_path, content_path,
                                   transform, batch_size, shuffle, num_workers)

    if mode == 'train':
        # In the 'train' mode, both 'style' and 'content' loaders are defined.
        content_loader, style_loader = image_dataset_loaders
    else:  # The model is in 'eval' mode. Only the 'style' loader is defined.
        style_loader = image_dataset_loaders

    # Create the style transfer model.
    style_transfer, optimizer = style_transfer_model(feature_layers,
                                               style_layers, len(style_loader))

    # Load a style transfer model checkpoint file.
    # If the path is not specified, new checkpoint data are initialized.
    style_transfer, optimizer, iteration, iter_loss = load_checkpoint(
              checkpoint_path, device, style_loader, style_transfer, optimizer)

    # Create the style transfer model dictionary.
    # With keys are used in both 'train' and 'eval' modes.
    model = {
        'style_transfer': style_transfer,
        'device': device,
    }

    if mode == 'train':
        # Update the model dictionary with the 'train' mode keys.
        model.update({
            'style_loader': style_loader,
            'content_loader': content_loader,
            'optimizer': optimizer,
            'iteration': iteration,
            'iter_loss': iter_loss,
            'feature_layers': feature_layers,
            'style_layers': style_layers,
            'style_weight': style_weight,
        })
    else:  # The model is in 'eval' mode. Set the model to evaluation mode.
        model['style_transfer'].eval()

    return model
