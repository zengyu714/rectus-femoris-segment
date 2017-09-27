import time
import glob

import torch
import numpy as np

import torch.utils.data as data

from scipy.misc import imread
from skimage.exposure import adjust_gamma
from skimage.transform import AffineTransform, warp, resize


def _normalize(im):
    """Normalize image's intensity to range [0, 1], for following image processing"""

    # `im` with dtype float64
    _max, _min = np.max(im), np.min(im)
    return (im - _min) / (_max - _min)


def random_gamma(image):
    """Adjust images' intensity"""

    # `inputs` has shape [height, width] with value in [0, 1].
    gamma = np.random.uniform(low=0.8, high=1.2)
    return adjust_gamma(image, gamma)


def random_hflip(image, label, u=0.5):
    if np.random.random() < u:
        image = np.fliplr(image)  # may be wrong, take a look on axis
        label = np.fliplr(label)
    return image, label


def random_affine(image, label,
                  scale_limit=(0.9, 1.1), trans_limit=(-0.0625, 0.0625), shear_limit=(-3, 3), rot_limit=(-3, 3)):
    """Random affine transformation"""

    rotation = np.random.uniform(*np.deg2rad(rot_limit))
    shear = np.random.uniform(*np.deg2rad(shear_limit))
    translation = np.random.uniform(*np.multiply(image.shape[:-1], trans_limit), size=2)

    tform = AffineTransform(scale=scale_limit, rotation=rotation, shear=shear, translation=translation)
    return [warp(item, tform, mode='reflect', preserve_range=True) for item in [image, label]]


class RectusFemorisDataset(data.Dataset):
    def __init__(self, mode='train', image_size=(256, 256)):
        """Assume dataset is in directory '.data/rf_SX_image/' and '.data/rf_SX_label',
        X ∈ {base, 0, 1, 2, 13, 15, 16, 18, 19, 24}
        """
        super(RectusFemorisDataset, self).__init__()
        self.image_size = image_size

        self.data_path = glob.glob('data/*_label/*.bmp')  # 270 training images
        np.random.shuffle(self.data_path)

        split = int(len(self.data_path) * 0.1)
        self.train_path, self.test_path = self.data_path[:-split], self.data_path[-split:]

        self.label_path = eval('self.{}_path'.format(mode))
        self.image_path = [p.replace('label', 'image') for p in self.label_path]

    def __getitem__(self, index):
        # Set random seed for random augment
        np.random.seed(int(time.time()))

        # Load gray image, `F` (32-bit floating point pixels)
        im, lb = [resize(imread(p[index], mode='F'), self.image_size, preserve_range=True)
                  for p in [self.image_path, self.label_path]]

        # Transform
        im = random_gamma(_normalize(im))
        im, lb = random_affine(im, lb)

        # [batch_size, channel, height, width]
        lb = (lb > 127).astype(np.float32)
        im, lb = [item[np.newaxis, ...] for item in [im, lb]]

        # Convert to Tensor
        return [torch.from_numpy(item) for item in [im, lb]]

    def __len__(self):
        return len(self.image_path)


class DeployRectusFemoris(data.Dataset):
    def __init__(self, mode='total', image_size=(256, 256)):
        """
        Arguments:
            image_size: same with training input size
            mode:
                total: 2540 images
                train: 270  images
                infer: 2270 images
        """
        self.mode = mode
        self.image_size = image_size

        total_image = glob.glob('data/*_image/*.bmp')
        train_image = [p.replace('label', 'image') for p in glob.glob('data/*_label/*.bmp')]  #
        infer_image = list(set(total_image) - set(train_image))
        self.images_path = eval('{}_image'.format(mode))

    def __getitem__(self, idx):
        path = self.images_path[idx]
        image = imread(path, mode='F')
        image = resize(image, self.image_size, preserve_range=True)
        image = _normalize(image)
        return torch.from_numpy(np.expand_dims(image, 0)), path

    def __len__(self):
        return len(self.images_path)


def _test_data_loader(dataset=RectusFemorisDataset()):
    vis = visdom.Visdom()

    training_data_loader = data.DataLoader(dataset=dataset, num_workers=4, batch_size=4, shuffle=True)
    im, lb = next(iter(training_data_loader))
    if isinstance(lb, torch.FloatTensor):
        print(im.size(), 'image')
        print(lb.size(), 'label')
        vis.images(im.numpy(), opts=dict(title='Random selected image', caption='Shape: {}'.format(im.size())))
        vis.images(lb.numpy(), opts=dict(title='Random selected label', caption='Shape: {}'.format(lb.size())))
    else:
        print(im.size(), 'image')
        print(lb, 'image path')
        vis.images(im.numpy(), opts=dict(title='Random selected image', caption='Shape: {}'.format(im.size())))


if __name__ == '__main__':
    import visdom

    # _test_data_loader(dataset=DeployRectusFemoris(mode='infer'))
    _test_data_loader()
