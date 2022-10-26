'''Define the style transfer network training function.'''
from pathlib import Path
from time import time

from .utils import create_checkpoint

from ..model import style_transfer_loss


def train(
    model,
    save_dir_path,
    save_minutes=10,
    save_iterations=None,
    max_iterations=25000,
    verbose=True,
    verbose_loss_iters=1
):
    '''
    Train the style transfer network.

    Args:
        model: dict. Style transfer network built model, as returned by
            `build_model.build_model` function.
        save_dir_path: str. Checkpoint save path (with filename).
        save_minutes: int. Time (in minutes) to elapse for checkpoint creation.
            When not specified (None), the feature is disabled.
        save_iterations: int. Iterations to elapse for checkpoint creation.
        max_iterations: int. Network training iterations.
        verbose: bool. Whether to display training info or not.
        verbose_loss_iters: int. Iterations to elapse for detailed styles' losses
            display. Feature disabled when not specified (zero or negative value).

    Returns:
        None.
    '''

    # Get the style/content image dataset loaders.
    content_loader = model['content_loader']
    style_loader = model['style_loader']

    # Get the style transfer network, its optimizer and the device it is loaded on.
    style_transfer = model['style_transfer']
    optimizer = model['optimizer']
    device = model['device']

    # Get the model iteration index value and its training logs.
    iteration = model['iteration']
    iter_loss = model['iter_loss']

    feature_layers = model['feature_layers']
    style_layers = model['style_layers']
    style_weight = model['style_weight']

    # Define an iterator over content images dataset.
    # An images' batch (of size "batch_size") will be returned on each iteration.
    iter_content_train = iter(content_loader)

    # If the checkpoints save directory does not exist, create it.
    Path(save_dir_path).mkdir(parents=True, exist_ok=True)

    if iteration == 0:
        print('No checkpoint file specified, train a new model.\n')
    else:
        print(f'Resume model training from the iteration: {iteration}.\n')

    loop_time_begin = time()

    # Define the training loop.
    while True:
        iter_time_begin = time()

        try:
            # Get a content images' batch of size "batch_size".
            # "curr_content_batch" is a tensor of shape (batch_size, C, W, H)
            # where C, W, H are images' Channels, Width, Height respectively.
            curr_content_batch = iter_content_train.next().to(device=device)
        except StopIteration:
            # If no batch is available, then all content images have been processed.
            # That is, iterating over all the dataset's images marks a new epoch.
            if verbose:
                print('• New epoch reached.')

            # Start a new epoch by defining a new iterator over content images dataset.
            iter_content_train = iter(content_loader)
            curr_content_batch = iter_content_train.next().to(device=device)

        # Define current iteration stats (styles' losses and training time).
        iter_style_loss_dict = {}

        # For the current content images' batch, iterate over all model's styles.
        for style_num, curr_style in enumerate(style_loader):
            curr_style = curr_style.unsqueeze(0)
            curr_style = curr_style.to(device=device)

            # The style transfer network pipeline is:
            # - Input: Content and the style images.
            # - Process: Get the stylized content images (style transfer module's
            #       output), and pass them (along with the input content/style images)
            #       to the Perceptual Network (a pre-trained VGG-16 model).
            # - Output: VGG-16 intermediate layers' weights.
            inter_layers = style_transfer(curr_content_batch, style_num, curr_style)
            # Compute the style transfer loss.
            loss = style_transfer_loss(inter_layers, feature_layers, style_layers,
                                    style_weight=style_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Add current style loss to the stats dict.
            iter_style_loss_dict[style_num] = loss.item()

        current_time = time()

        iteration += 1
        iter_style_loss_dict['time'] = current_time - iter_time_begin

        # Add the current iteration stats to the global training stats.
        iter_loss.append(iter_style_loss_dict)

        if verbose:  # If the verbose is enabled.
            # Display current iteration time.
            print(f'Iteration: {iteration} | '
                  f'Time: {time()-iter_time_begin:.2f} sec.')

            if (verbose_loss_iters > 0) and (iteration % verbose_loss_iters == 0):
                # For each style, display its loss for the current iteration.
                for style_num, style_name in enumerate(style_loader.root_images):
                    style_loss = iter_style_loss_dict[style_num]
                    print(f'  » Style {style_num:2d} ({style_name}): {style_loss:.3f}')

        # Get elapsed minutes since "loop_time_begin".
        loop_minutes_diff = (current_time - loop_time_begin) // 60

        # Evaluate the checkpoint creation conditions.
        # Evaluate the 'iteration' condition: Check whether the specified
        # iterations have elapsed.
        save_iters_cond = (save_iterations is not None) and (iteration % save_iterations == 0)
        # Evaluate the 'time' condition: Check whether the specified time
        # (in minutes) has elapsed.
        save_time_cond = (save_minutes is not None) and (loop_minutes_diff >= save_minutes)
        # Evalute the 'maximum iterations' condition.
        max_iters_cond = (iteration == max_iterations)
        # A new checkpoint is created only if at least one of the three
        # previous conditions are True.
        if save_iters_cond or save_time_cond or max_iters_cond:
            checkpoint_path = save_dir_path / f'checkpoint_{iteration}.tar'

            create_checkpoint(checkpoint_path,
                            style_loader.root_images,
                            model_state_dict=style_transfer.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(),
                            iteration=iteration, iter_loss=iter_loss)

            if verbose:
                print(f'\n• A new checkpoint has been saved in: {checkpoint_path}\n')

            # Reset the 'iteration' condition start time.
            loop_time_begin = time()

            # If the specified maximum iteration has been reached, stop the training.
            if max_iters_cond:
                if verbose:
                    print(f'\nMaximum iteration ({max_iterations}) reached, '
                           'training finished.')
                break
