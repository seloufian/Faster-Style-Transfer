'''Define helper functions used in the style transfer model training and evaluation.'''
import torch
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from ..model import StyleTransferDataset, StyleTransferModule, center_crop


def create_checkpoint(save_path, styles_dataset_images, model_state_dict,
                      optimizer_state_dict, iteration, iter_loss):
    '''
    Create a style transfer model checkpoint file.

    Args:
        save_path: str. Save path (with filename).
        styles_dataset_images: list of str. Style images filenames. The order is important.
        model_state_dict: OrderedDict. Style transfer model's PyTorch state_dict.
        optimizer_state_dict: dict. Style transfer optimizer's PyTorch state_dict.
        iteration: int. Checkpoint's iteration index value.
        iter_loss: list of dict. Training losses/time for each iteration.
            A list of length `iteration`, where each element is a dict:
                <style_number>: 'int' keys with 'float' values. Style's loss.
                    Each <style_number> corresponds to a style of `styles_dataset_images`.
                time: float. Iteration training time (in seconds).

    Returns:
        None.
        The ckeckpoint will be saved in the given path.

    Raises:
        ValueError: If `iteration` value is different from `iter_loss` length.
    '''
    if iteration != len(iter_loss):
        raise ValueError(f"The 'iteration' value ({iteration}) is different "
                         f"from the 'iter_loss' length ({len(iter_loss)}).")

    checkpoint_dict = {
        'styles_dataset_images': styles_dataset_images,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'iteration': iteration,
        'iter_loss': iter_loss
    }

    torch.save(checkpoint_dict, save_path)



def opt_state_to_device(optimizer_state, device):
    '''
    Move optimizer's PyTorch state_dict tensors to the given device.

    Args:
        optimizer_state: dict. Optimizer's PyTorch state_dict.
        device: str. Device to which the state_dict tensors will be moved.

    Returns:
        optimizer_state: dict. Updated state_dict.
    '''
    for state in optimizer_state.values():
        for key, val in state.items():
            # A "state_dict" contains multiple data types.
            # Only Tensors (PyTorch structure) are considered.
            if isinstance(val, torch.Tensor):
                state[key] = val.to(device)

    return optimizer_state



def dataset_loader(style_path, content_path=None, transform=True,
                   batch_size=16, shuffle=True, num_workers=1):
    '''
    Create the style/content image dataset PyTorch loaders.

    Args:
        style_path: str. Style images directory path.
            Accepted image extensions are: 'png' and 'jpg'.
        content_path: str, optional. Content images directory path.
            Defaults to 'None' (not specified).
        transform: bool. Whether to transform (center crop) the images or not.
        batch_size: int. Batch size, preferably a power of 2.
        shuffle: bool. Whether to shuffle the content images or not.
        num_workers: int. Number of parallel workers for content images loading.
            The recommended value is the number of CPU cores.

    Returns:
        content_loader: DataLoader, optional (if `content_path` specified).
            Content images loader.
        style_loader: StyleTransferDataset. Style images dataset.
    '''
    if content_path is not None:
        transform = center_crop if transform else None

        content_loader = StyleTransferDataset(content_path, transform=transform)
        content_loader = DataLoader(content_loader, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=num_workers)

    style_loader = StyleTransferDataset(style_path, transform=transform)

    return style_loader if (content_path is None) else (content_loader, style_loader)



def style_transfer_model(feature_layers, style_layers, num_styles):
    '''
    Create the style transfer model.

    Args:
        feature_layers: list of int. VGG-16 layers for the content features extraction.
        style_layers: list of int. VGG-16 layers for the style features extraction.
        num_styles: int. Number of model's styles.

    Returns:
        style_transfer: StyleTransferModule. Style transfer model.
        optimizer: Adam. Model's Adam optimizer.
    '''
    inter_layers_num = list(set(feature_layers + style_layers))
    style_transfer = StyleTransferModule(inter_layers_num, num_styles=num_styles)

    optimizer = torch.optim.Adam(style_transfer.parameters())

    return style_transfer, optimizer



def load_checkpoint(checkpoint_path, device, style_loader,
                    style_transfer, optimizer):
    '''
    Load a style transfer model checkpoint file.

    Args:
        checkpoint_path: str. Load path.
            If not specified (None), a new checkpoint data are initialized.
        device: str. Device on which the model and its optimizer will be moved.
        style_loader: StyleTransferDataset. Style images dataset.
        style_transfer: StyleTransferModule. Style transfer model.
        optimizer: Model's optimizer.

    Returns:
        style_transfer: StyleTransferModule. Synchronized model (updated weights).
        optimizer: Synchronized optimizer (updated weights).
        iteration: int. Model iteration index value.
            Defaults to 0, if no `checkpoint_path` specified.
        iter_loss: dict. Training losses/time for each iteration.
            Defaults to [], if no `checkpoint_path` specified.

    Raises:
        ValueError: If the dataset's style images are different from the checkpoint's ones.
    '''
    # Check whether the checkpoint path is specified.
    if checkpoint_path is not None:
        # Load the checkpoint file, and move it to the given device.
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get style images filenames list.
        styles_dataset_images = checkpoint['styles_dataset_images']

        # The dataset's style images must be similar to the checkpoint's ones.
        # That is, used styles in the model are referenced -only- by their index.
        # So, we have to insure that model's (checkpoint) styles are the same as
        # the ones defined in the dataset.
        if set(style_loader.root_images) != set(styles_dataset_images):
            raise ValueError("The dataset's style images are different from "
                             "the checkpoint's ones.")

        # Preserving style images' order is important.
        # For that, the dataset images are reorganized to have a similar
        # order to the checkpoint ones.
        # This operation can be done because we have already insured
        # the similarity between the two lists' items.
        style_loader.root_images = styles_dataset_images

        # Load the iteration index value.
        iteration = checkpoint['iteration']
        # Load the training losses/time for each iteration.
        iter_loss = checkpoint['iter_loss']

        # Load the style transfer model's state_dict.
        style_transfer.load_state_dict(checkpoint['model_state_dict'])

        # Load the style transfer optimizer's state_dict.
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Even that the checkpoint is moved to the specified "device",
        # optimizer's data are still on CPU (probably, a PyTorch issue).
        # So, the data has to be moved 'manually' to the given "device".
        optimizer.state = opt_state_to_device(optimizer.state, device=device)

    else:  # The checkpoint path is not specified.
        # Initialize new checkpoint data.
        iteration, iter_loss = 0, []

    style_transfer = style_transfer.to(device=device)

    return style_transfer, optimizer, iteration, iter_loss



def visualize_images(sides, rows, size_factor=3, spacing=True, save_path=None):
    '''
    Visualize images in a grid.
    Used for both, 2 and 4 weighted style transfer inference.

    Args:
        sides: list of `PIL` images.
            List of images to be placed on the sides of the grid.
        rows: list of "list of `PIL` images".
            List of rows, where each row is a list of images.
        size_factor: int. Grid images size factor.
        spacing: bool. Whether to add spacing between the images or not.
        save_path. str. Path to save the visualization.
            Defaults to None (do not save the visualization).

    Returns:
        plt: matplotlib.pyplot. Figure containing the visualization.

    Raises:
        ValueError:
            If the rows contain no columns.
            If the number of sides is not either 2 or 4.
            If there is more than one row (for the 2-sided grid).
            If the number of rows and columns is not equal (for the 4-sided grid).
    '''
    if len(rows[0]) == 0:
        raise ValueError('There must be at least 1 column.')

    if len(sides) not in [2, 4]:
        raise ValueError(f'The number of sides must be 2 or 4. '
                         f'Given {len(sides)}.')

    if (len(sides) == 2) and (len(rows) != 1):
        raise ValueError(f'For the 2-sided grid, there must be 1 row. '
                         f'Given {len(rows)}.')

    if len(sides) == 4:
        if not all([len(rows) == len(row) for row in rows]):
            raise ValueError(f'For the 4-sided grid, the number of rows '
                              'and columns must be equal.')

    # Get the number of rows and columns.
    num_rows = len(rows)
    num_cols = len(rows[0]) + 2  # Added '2' for the "sides".

    # Define the figure size.
    fig_size = num_cols * size_factor

    # Create the figure: a grid of ("num_rows" x "num_cols") subplots.
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols,
                           figsize=(fig_size, fig_size))

    # For the 4-sided grid, the "ax" is a 2D array (rows x columns).
    # However, for the 2-sided grid, the "ax" is a 1D array.
    # Given that, add a new dimension to match the 4-sided "ax" shape.
    if len(sides) == 2:
        ax = ax[np.newaxis, :]

    # Add the "rows" images to the grid.
    for i in range(num_rows):  # Loop over the rows.
        curr_row = rows[i]  # Get the current row.
        for j in range(num_cols):  # Loop over the current row's columns.
            ax[i, j].axis('off')  # Turn 'off' the axis display.
            ax[i, j].set_aspect('equal')

            # The first (0) and the last (cols-1) columns represent the "sides".
            # The in-between columns (from 1 to 'cols-2') represent the "rows".
            if 1 <= j <= num_cols-2:
                ax[i, j].imshow(curr_row[j-1])

    # Define the sides' images grid coordinates.
    # For the 2-sided grid, the side images comes on the 'left' and 'right'.
    sides_coords = [(0, 0), (0, -1)]
    if len(sides) == 4:
        # For the 4-sided grid, the side images comes on the:
        # Upper-left, upper-right, lower-left, lower-right.
        sides_coords += [(-1, 0), (-1, -1)]

    # Add the sides' images to the grid.
    for idx, side_coord in enumerate(sides_coords):
        ax[side_coord].set_aspect('equal')

        # Turn 'on' the axis display, but disable the X and Y axis ticks.
        # Used to add borders.
        ax[side_coord].axis('on')
        ax[side_coord].set_xticks([])
        ax[side_coord].set_yticks([])

        ax[side_coord].imshow(sides[idx])

        # Add a black border to the sides' images.
        [spine.set_linewidth(1.7) for spine in ax[side_coord].spines.values()]

    # Add some spacing, if specified, between the grid's images.
    # The parameters' values were chosen experimentally.
    if spacing:
        hspace, wspace = -0.66, 0.05
    else:
        hspace, wspace = -0.663, 0

    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    # Save the visualization (if specified) on a transparent background.
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', transparent=True)

    # Close the figure so that it does not display automatically.
    plt.close()

    return fig
