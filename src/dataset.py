import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.io import read_image

import os
from enum import Enum

class DSubset(Enum):
    """
    Indicator for whether ImageDataset should contain training
    or test instances.
    """
    TRAIN = 1
    TEST = 2

class Label(Enum):
    """
    Indicator for whether ImageDataset should have instance labels be 
    the object within the image OR whether or not the image was synthetically
    generated.
    """
    OBJECT = 1 # Target label is object in the image (10 options)
    REAL_OR_SYNTHETIC = 2 # Target label is whether the image is real or synthetically generated (2 options)

class ImageDataset(Dataset):
    """
    A custom dataset class designed to contain and produce examples originating
    from the CIFAKE (https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/data)
    dataset.
    """
    def __init__(self, split: DSubset, label_type: Label, transform = None, target_transform = None):
        """
        Initializer for ImageDataset.

        Args:
            split -- An indicator for the type of examples contained, see DSubset enum.
            label_type -- An indicator for the type of instance label, see Label enum.
            transform -- The torch transform to be exacted on the image.
            target_transform -- The torch transform to be exacted on the label.
        """

        assert isinstance(split, DSubset)
        assert isinstance(label_type, Label)
        
        self.split = 'train' if split == DSubset.TRAIN else 'test'
        self.label_type = label_type

        self.listdirs = {
            'REAL': os.listdir(f'../data/{self.split}/REAL/'),
            'FAKE': os.listdir(f'../data/{self.split}/FAKE/'),
        }

        self.num_real_instances = len(self.listdirs['REAL'])
        self.num_synthetic_instances = len(self.listdirs['FAKE'])

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Provides the number of datapoints contained within the ImageDatset.

        Examples originate from the REAL and FAKE subdirectories, making their
        sum for the designated DSubset the total dataset length.
        """
        return self.num_real_instances + self.num_synthetic_instances

    def __getitem__(self, index):
        """
        Produces the item from the ImageDataset specified by index.

        Args:
            index -- Indicates which instance to produce.

        Returns:
            X -- Image features (i.e. pixels)
            y -- Image label
        """

        if index < 0 or index >= self.__len__():
            raise IndexError(f'Provided index out of range of number of files available in {self.split} split.')
        
        # Locate image
        image_type = 'REAL' if index < self.num_real_instances else 'FAKE'
        folder_path = f'../data/{self.split}/{image_type}/'

        # If image is fake, index into FAKE folder must be offset by number
        # of real instances to ensure proper zero-indexing.
        if image_type == 'FAKE':
            index -= self.num_real_instances

        image_name = self.listdirs[image_type][index]
        image_path = folder_path + image_name

        # Load image
        image = read_image(image_path)
        image = image.float() # convert from uint8 to float32
        image /= 255.0 # normalize to [0,1] scale from [0,255]

        label = None
        
        if self.label_type == Label.OBJECT:
            open_paren_index = image_name.find('(')
            closed_paren_index = image_name.find(')')
            label = int(image_name[open_paren_index+1:closed_paren_index]) if open_paren_index != -1 else 1
        elif self.label_type == Label.REAL_OR_SYNTHETIC:
            label = 0 if image_type == 'REAL' else 1 # (0 = Real, 1 = Synthetic)
        else:
            raise ValueError(f'No proper label type provided, was {self.label_type}')
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def get_channel_means_stdevs(dataloader: DataLoader, num_channels = 1, verbose = False) -> tuple[tuple[float], tuple[float]]:
    """
    Given the dataset contained within the dataloader, produces the pixel
    value means and standard deviations for each channel.

    Note:
        - torch.mean was considered, but torch.sum was used in tandem with
        the growing total_pixels to ensure that the final dataloader batch
        (often of a size smaller than batch_size) is properly averaged into
        the final result. Using torch.mean when including this last batch
        would overly weight those examples when channel_means = mean(batch_mean)
        in this scenario.

    Args:
        dataloader -- Links to the dataset containing the relevant images.
        num_channels -- The number of channels to be averaged into. Must be
        equal to the number of channels in the images in dataloader.
        verbose -- Indicates whether or not to print image process counts.

    Returns:
        tuple_1 -- pixel value means for each channel
        tuple_2 -- pixel value standard deviations for each channel
    """

    cumulative_sum = torch.zeros(num_channels, dtype = torch.float32) # RGB
    cumulative_squared_sum = torch.zeros(num_channels, dtype = torch.float32) # RGB
    total_pixels = 0

    for i, (X, _) in enumerate(dataloader):

        # Note: dim (0, 2, 3) specifies that which to REDUCE over (i.e. all pixels averaged into 3 channels)
        cumulative_sum += torch.sum(X, dim = (0, 2, 3), dtype = torch.float32)
        cumulative_squared_sum += torch.sum(X**2, dim = (0, 2, 3), dtype = torch.float32)

        total_pixels += X.size(0) * X.size(2) * X.size(3)

        if verbose and (i+1) % 100 == 0:
            print(f'Processed {(i+1) * dataloader.batch_size} total images.')

    channel_means = cumulative_sum / total_pixels
    channel_variances = (cumulative_squared_sum / total_pixels) - channel_means**2 # V[X] = E[X^2] - E[X]^2
    channel_stdevs = torch.sqrt(channel_variances)

    return tuple(channel_means.tolist()), tuple(channel_stdevs.tolist())