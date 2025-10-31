# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

import random
import PIL
from PIL import ImageFilter


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from timm.data import create_transform


_GLOBAL_SEED = 0
logger = getLogger()


def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    validation=False,
    supervised=False,
    normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
):
    logger.info("making imagenet data transforms")

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        return color_distort

    transform_list = []
    if validation:
        transform_list += [
            transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC)
        ]  # to maintain same ratio w.r.t. 224 images
        transform_list += [transforms.CenterCrop((224, 224))]
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize(normalization[0], normalization[1])]
        transform = transforms.Compose(transform_list)
        return transform

    if supervised:
        # -- Borrowed from MAE
        transform = create_transform(
            input_size=crop_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            mean=normalization[0],
            std=normalization[1],
        )
        return transform

    transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class _BaseSuperResCrop(nn.Module):
    """
    Abstract base class for super-resolution cropping transforms.

    The core forward pass logic is shared:
    1. A subclass implements `get_crop_params` to determine the starting coordinates.
    2. The `forward` method uses these coordinates to perform the synchronized
       cropping of the low-resolution input and high-resolution target.
    """

    def __init__(
        self,
        input_patch_size: int,
        output_patch_size: int,
        scale_factor: int = 6,
        len_seq_predict: int = 32,
    ):
        super().__init__()
        self.input_patch_size = input_patch_size
        self.output_patch_size = output_patch_size
        self.scale_factor = scale_factor
        self.len_seq_predict = len_seq_predict

    def get_crop_params(self, sample: tuple) -> tuple[int, int]:
        raise NotImplementedError

    def forward(self, sample: tuple) -> tuple:
        # x, y, metadata = sample
        x, y = sample
        lr_h, lr_w = x.shape[-2:]

        y_hr_start, x_hr_start = self.get_crop_params(y)

        lr_center_x = (x_hr_start + self.output_patch_size // 2) / self.scale_factor
        lr_center_y = (y_hr_start + self.output_patch_size // 2) / self.scale_factor
        y_lr_start = max(0, round(lr_center_y - self.input_patch_size / 2))
        x_lr_start = max(0, round(lr_center_x - self.input_patch_size / 2))

        if y_lr_start + self.input_patch_size > lr_h:
            y_lr_start = lr_h - self.input_patch_size
        if x_lr_start + self.input_patch_size > lr_w:
            x_lr_start = lr_w - self.input_patch_size

        x_cropped = x[
            ...,
            y_lr_start : y_lr_start + self.input_patch_size,
            x_lr_start : x_lr_start + self.input_patch_size,
        ]

        # Handle placeholders, creating tensors on the same device as the input.
        placeholder_shape = (
            1,
            self.len_seq_predict,
            self.output_patch_size,
            self.output_patch_size,
        )

        if y.numel() > 0:
            y_cropped = y[
                ...,
                y_hr_start : y_hr_start + self.output_patch_size,
                x_hr_start : x_hr_start + self.output_patch_size,
            ]
        else:
            y_cropped = torch.zeros(placeholder_shape, dtype=x.dtype, device=x.device)

        # mask = metadata.get("target", {}).get("mask")

        # if mask is not None and mask.numel() > 0:
        #     mask_cropped = mask[
        #         ...,
        #         y_hr_start : y_hr_start + self.output_patch_size,
        #         x_hr_start : x_hr_start + self.output_patch_size,
        #     ]
        #     metadata["target"]["mask"] = mask_cropped
        # else:
        #     metadata["target"]["mask"] = torch.zeros(
        #         placeholder_shape, dtype=x.dtype, device=x.device
        #     )

        # return x_cropped, y_cropped, metadata
        return x_cropped, y_cropped


class RandomSuperResCrop(_BaseSuperResCrop):
    """
    Performs a random crop, with an option to bias sampling towards rainy areas.
    """

    def __init__(
        self,
        input_patch_size: int,
        output_patch_size: int,
        scale_factor: int = 6,
        rain_sampling_p: float = 0.0,
        rain_sampling_threshold: float = 0.1,
    ):
        super().__init__(input_patch_size, output_patch_size, scale_factor)
        self.rain_sampling_p = rain_sampling_p
        self.rain_sampling_threshold = rain_sampling_threshold

    def get_crop_params(self, y: torch.Tensor) -> tuple[int, int]:
        hr_h, hr_w = y.shape[-2:]

        use_rain_sampling = (
            random.random() < self.rain_sampling_p
            and (y > self.rain_sampling_threshold).any()
        )

        if use_rain_sampling:
            # Create a 2D spatial mask of rainy pixels. Assumes y has a layout like
            # [C, T, H, W] or [T, H, W]. It collapses non-spatial leading dimensions.
            is_rainy_mask = y > self.rain_sampling_threshold
            while is_rainy_mask.ndim > 2:
                is_rainy_mask = is_rainy_mask.any(dim=0)

            rainy_indices = is_rainy_mask.nonzero()  # Shape: [num_rainy, 2]

            if rainy_indices.shape[0] > 0:
                # Select a random rainy pixel's (y, x) spatial index
                random_idx = torch.randint(0, len(rainy_indices), (1,)).item()
                center_y, center_x = rainy_indices[random_idx]

                y_hr_start = max(0, int(center_y) - self.output_patch_size // 2)
                x_hr_start = max(0, int(center_x) - self.output_patch_size // 2)
            else:
                use_rain_sampling = False  # Fallback if no rainy pixels found

        if not use_rain_sampling:
            # Perform a simple random crop
            y_hr_start = torch.randint(
                0, hr_h - self.output_patch_size + 1, (1,)
            ).item()
            x_hr_start = torch.randint(
                0, hr_w - self.output_patch_size + 1, (1,)
            ).item()

        # Ensure final coordinates are within bounds
        y_hr_start = min(y_hr_start, hr_h - self.output_patch_size)
        x_hr_start = min(x_hr_start, hr_w - self.output_patch_size)

        return y_hr_start, x_hr_start


class CenterSuperResCrop(_BaseSuperResCrop):
    """
    Performs a deterministic center crop for validation and testing.
    """

    def get_crop_params(self, y: torch.Tensor) -> tuple[int, int]:
        hr_h, hr_w = y.shape[-2:]

        # Calculate top-left coordinates for a crop centered in the HR image
        y_hr_start = hr_h // 2 - self.output_patch_size // 2
        x_hr_start = hr_w // 2 - self.output_patch_size // 2

        return y_hr_start, x_hr_start


class PredictionCrop(_BaseSuperResCrop):
    """
    Performs a deterministic crop for prediction based on coordinates
    provided in the sample's metadata dictionary.
    """

    def get_crop_params(self, sample: tuple) -> tuple[int, int]:
        metadata = sample[2]
        coords = metadata.get("coords", {})

        try:
            # These coordinates are from the test CSV file
            y_hr_start = int(coords["y-top-left"])
            x_hr_start = int(coords["x-top-left"])
        except KeyError as e:
            raise KeyError(
                f"Prediction coordinates not found in metadata. "
                f"Missing key: {e}. Available keys: {list(coords.keys())}"
            ) from e

        return y_hr_start, x_hr_start


class ToTensor(nn.Module):
    """
    Converts numpy arrays in a sample to PyTorch tensors.

    This transform is designed to be placed in a pipeline before other
    tensor-based augmentations. It handles the (input, output, metadata)
    tuple format, including the nested mask tensor.
    """

    def forward(self, sample: tuple) -> tuple:
        """
        Args:
            sample (tuple): A tuple containing (input_data, target_data, metadata).
                            input_data, target_data, and the mask are expected
                            to be numpy arrays.

        Returns:
            tuple: A tuple where all relevant numpy arrays are converted to
                   torch.FloatTensor.
        """
        input_data, target_data, metadata = sample

        # .copy() is used to avoid a potential warning about non-writable tensors
        # if the numpy array is read-only.
        input_tensor = torch.from_numpy(input_data.copy()).float()
        target_tensor = torch.from_numpy(target_data.copy()).float()

        # Also convert the mask within the metadata dictionary to a tensor
        if "target" in metadata and "mask" in metadata["target"]:
            mask_np = metadata["target"]["mask"]
            if mask_np is not None and isinstance(mask_np, torch.Tensor) is False:
                metadata["target"]["mask"] = torch.from_numpy(mask_np.copy()).float()

        return input_tensor, target_tensor, metadata


class SynchronizedAugmentations(nn.Module):
    """
    Applies the same random spatial augmentations to both input and target tensors.

    This ensures that the spatial relationship between the satellite context and
    the rainfall target is maintained after transformations like flipping.
    """

    def __init__(self, p_vertical_flip: float = 0.5, p_horizontal_flip: float = 0.5):
        super().__init__()
        self.p_vertical_flip = p_vertical_flip
        self.p_horizontal_flip = p_horizontal_flip

    def forward(self, sample: tuple) -> tuple:
        x, y, metadata = sample

        # Random Horizontal Flip
        if torch.rand(1) < self.p_horizontal_flip:
            x = F.hflip(x)
            y = F.hflip(y)
            if "target" in metadata and "mask" in metadata["target"]:
                metadata["target"]["mask"] = F.hflip(metadata["target"]["mask"])

        # Random Vertical Flip
        if torch.rand(1) < self.p_vertical_flip:
            x = F.vflip(x)
            y = F.vflip(y)
            if "target" in metadata and "mask" in metadata["target"]:
                metadata["target"]["mask"] = F.vflip(metadata["target"]["mask"])

        return x, y, metadata


class PadChannels(nn.Module):
    """
    Pads the channel dimension of a tensor to a target size.
    Assumes the input tensor has channels as the first dimension (C, ...).
    """

    def __init__(self, target_channels: int, pad_value: float = 0.0):
        super().__init__()
        self.target_channels = target_channels
        self.pad_value = pad_value

    def forward(self, sample: tuple) -> tuple:
        x, y, metadata = sample

        current_channels = x.shape[0]
        if current_channels >= self.target_channels:
            return sample

        pad_needed = self.target_channels - current_channels

        pad_shape = list(x.shape)
        pad_shape[0] = pad_needed

        padding_tensor = torch.full(
            pad_shape, self.pad_value, dtype=x.dtype, device=x.device
        )

        # Concatenate along the channel dimension (dim=0)
        x_padded = torch.cat([x, padding_tensor], dim=0)

        return x_padded, y, metadata


class DeterministicCrop(_BaseSuperResCrop):
    """
    Performs a deterministic crop for prediction based on coordinates
    provided in the sample's metadata dictionary.
    """

    def get_crop_params(self, sample: tuple) -> tuple[int, int]:
        metadata = sample[2]
        coords = metadata["coords"]
        y_hr_start = int(coords["y-top-left"])
        x_hr_start = int(coords["x-top-left"])
        return y_hr_start, x_hr_start
