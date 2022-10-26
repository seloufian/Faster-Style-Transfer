'''Define the dataset for the style transfer module.'''
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import io, transforms


class StyleTransferDataset(Dataset):
    '''
    Dataset for the style transfer module.

    The dataset is composed of images only (without targets/classes), for
    both the content (large number of samples) and style images (few samples).
    '''
    def __init__(self, root_dir, transform=None):
        '''
        Style transfer module dataset's constructor.

        Note: The only allowed images' extensions are 'png' and 'jpg'.

        Args:
            root_dir: str. Root directory path which contains the images.
            transform: function. Transformation to apply (if specified) to each image.
        '''
        self.root_dir = Path(root_dir)

        # Get the list of images' names contained in the specified root directory path.
        self.root_images = [path.name for path in self.root_dir.glob('*') \
                            if path.suffix in ['.jpg', '.png']]

        self.transform = transform

    def __len__(self):
        '''Get the number of samples (images) in the dataset.'''
        return len(self.root_images)

    def __getitem__(self, idx):
        '''Get a sample (image) from the dataset (specified by its index `idx`).'''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir / self.root_images[idx]

        sample = io.read_image(img_name)

        # If an image is in grayscale (it has only one channel), convert it to
        # RGB by duplicating its values on the three RGB channels.
        if sample.shape[0] == 1:
            sample = sample.repeat(3, 1, 1)

        if self.transform:
            sample = self.transform(sample)

        return sample


def center_crop(image, size=256, normalize=True):
    '''
    Center crop an image.

    Args:
        image: 3-D tensor of shape (C, W, H). The input image.
        size: int. The size to which the input image will be center cropped.
        normalize: bool. Normalize the image values (by clipping them to [0,1])
            or not (if they have already been normalized).

    Returns:
        A center cropped image, a 3-D tensor of shape (C, size, size).
    '''
    # Image normalization (values in [0,1] interval) is needed to center crop it.
    if normalize:
        image = image.to(dtype=torch.float32) / 255

    # Rescale the image's smallest side to "size" (with keeping its aspect ratio).
    # Process explanatory code:
    # scale = size / (W if (H > W) else H)
    # new_W, new_H = int(scale * W), int(scale * H)
    image = transforms.functional.resize(image, size)
    # Center crop the rescaled image, so its width and height will have a length of "size".
    crop = transforms.CenterCrop(size)
    image = crop(image)

    return image
