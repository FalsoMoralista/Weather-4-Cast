import random

import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class _BaseSuperResCrop(nn.Module):
    """
    Abstract base class for super-resolution cropping transforms.
    This must be called AFTER data has been converted to tensors.

    The core forward pass logic is shared:
    1. A subclass implements `get_crop_params` to determine the starting coordinates.
    2. The `forward` method uses these coordinates to perform the synchronized
       cropping of the low-resolution input and high-resolution target tensors.
    """

    def __init__(
        self,
        input_patch_size: int,
        output_patch_size: int,
        scale_factor: int = 6,
        len_seq_predict: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.input_patch_size = input_patch_size
        self.output_patch_size = output_patch_size
        self.scale_factor = scale_factor
        self.len_seq_predict = len_seq_predict

    def get_crop_params(self, sample: tuple) -> tuple[int, int]:
        raise NotImplementedError

    def forward(self, sample: tuple) -> tuple:
        x, y, _ = sample
        lr_h, lr_w = x.size()[-2:]

        y_hr_start, x_hr_start = self.get_crop_params(sample)

        # Calculate the low-resolution crop coordinates based on the high-resolution crop
        lr_center_x = (x_hr_start + self.output_patch_size / 2) / self.scale_factor
        lr_center_y = (y_hr_start + self.output_patch_size / 2) / self.scale_factor
        y_lr_start = max(0, round(lr_center_y - self.input_patch_size / 2))
        x_lr_start = max(0, round(lr_center_x - self.input_patch_size / 2))

        # Ensure the low-resolution crop is within bounds
        if y_lr_start + self.input_patch_size > lr_h:
            y_lr_start = lr_h - self.input_patch_size
        if x_lr_start + self.input_patch_size > lr_w:
            x_lr_start = lr_w - self.input_patch_size

        # Perform the crop on the low-resolution input
        x_cropped = x[
            ...,
            y_lr_start : y_lr_start + self.input_patch_size,
            x_lr_start : x_lr_start + self.input_patch_size,
        ]

        # Define shape for placeholder tensors if data is missing
        placeholder_shape = (
            1,
            self.len_seq_predict,
            self.output_patch_size,
            self.output_patch_size,
        )

        # Crop the high-resolution target or create a zero tensor placeholder
        if y.numel() > 0:
            y_cropped = y[
                ...,
                y_hr_start : y_hr_start + self.output_patch_size,
                x_hr_start : x_hr_start + self.output_patch_size,
            ]
        else:
            y_cropped = torch.zeros(placeholder_shape, dtype=x.dtype, device=x.device)

        return x_cropped, y_cropped, _


class RandomSuperResCrop(_BaseSuperResCrop):
    """
    Performs a random crop on tensors, with an option to bias sampling towards rainy areas.
    """

    def __init__(
        self,
        input_patch_size: int,
        output_patch_size: int,
        scale_factor: int = 6,
        len_seq_predict: int = 16,
        rain_sampling_p: float = 0.0,
        rain_sampling_threshold: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            input_patch_size, output_patch_size, scale_factor, len_seq_predict
        )
        self.rain_sampling_p = rain_sampling_p
        self.rain_sampling_threshold = rain_sampling_threshold

    def get_crop_params(self, sample: tuple) -> tuple[int, int]:
        y = sample[1]
        hr_h, hr_w = y.size()[-2:]

        # Determine if rain-biased sampling should be used
        use_rain_sampling = (
            random.random() < self.rain_sampling_p
            and y.numel() > 0
            and (y > self.rain_sampling_threshold).any()
        )

        if use_rain_sampling:
            # Find all pixels where rainfall exceeds the threshold
            is_rainy_mask = y > self.rain_sampling_threshold
            dims_to_collapse = tuple(range(y.ndim - 2))
            if dims_to_collapse:
                is_rainy_mask = torch.any(is_rainy_mask, dim=dims_to_collapse)

            rainy_indices = torch.argwhere(is_rainy_mask)

            # If rainy pixels are found, pick one to be the center of the crop
            if rainy_indices.size(0) > 0:
                rand_idx = torch.randint(0, rainy_indices.size(0), (1,)).item()
                center_y, center_x = rainy_indices[rand_idx]

                y_hr_start = max(0, int(center_y) - self.output_patch_size // 2)
                x_hr_start = max(0, int(center_x) - self.output_patch_size // 2)
            else:
                use_rain_sampling = False  # Fallback to random if no pixels found

        # Default to a standard random crop
        if not use_rain_sampling:
            y_hr_start = torch.randint(
                0, hr_h - self.output_patch_size + 1, (1,)
            ).item()
            x_hr_start = torch.randint(
                0, hr_w - self.output_patch_size + 1, (1,)
            ).item()

        # Ensure the final crop coordinates are within image bounds
        y_hr_start = min(y_hr_start, hr_h - self.output_patch_size)
        x_hr_start = min(x_hr_start, hr_w - self.output_patch_size)

        return y_hr_start, x_hr_start


class CenterSuperResCrop(_BaseSuperResCrop):
    """
    Performs a deterministic center crop on tensors for validation and testing.
    """

    def get_crop_params(self, sample: tuple, **kwargs) -> tuple[int, int]:
        y = sample[1]
        if y.numel() == 0:
            raise ValueError("CenterSuperResCrop cannot be applied to an empty tensor.")

        hr_h, hr_w = y.size()[-2:]

        y_hr_start = (hr_h - self.output_patch_size) // 2
        x_hr_start = (hr_w - self.output_patch_size) // 2

        return y_hr_start, x_hr_start


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


class PadChannels(nn.Module):
    """
    Pads the channel dimension of a tensor to a target size.
    (This class was already using PyTorch and requires no changes.)
    """

    def __init__(self, target_channels: int, pad_value: float = 0.0):
        super().__init__()
        self.target_channels = target_channels
        self.pad_value = pad_value

    def forward(self, sample: tuple) -> tuple:
        x, y, metadata = sample

        current_channels = x.size(0)
        if current_channels >= self.target_channels:
            return sample

        pad_needed = self.target_channels - current_channels
        pad_shape = list(x.size())
        pad_shape[0] = pad_needed

        padding_tensor = torch.full(
            pad_shape, self.pad_value, dtype=x.dtype, device=x.device
        )
        x_padded = torch.cat([x, padding_tensor], dim=0)

        return x_padded, y, metadata


class SynchronizedAugmentations(nn.Module):
    """
    Applies the same random spatial augmentations to both input and target tensors.
    (This class was already using PyTorch and requires no changes.)
    """

    def __init__(self, p_vertical_flip: float = 0.5, p_horizontal_flip: float = 0.5):
        super().__init__()
        self.p_vertical_flip = p_vertical_flip
        self.p_horizontal_flip = p_horizontal_flip

    def forward(self, sample: tuple) -> tuple:
        x, y, metadata = sample

        if torch.rand(1) < self.p_horizontal_flip:
            x = F.hflip(x)
            y = F.hflip(y)
            if "target" in metadata and "mask" in metadata["target"]:
                metadata["target"]["mask"] = F.hflip(metadata["target"]["mask"])

        if torch.rand(1) < self.p_vertical_flip:
            x = F.vflip(x)
            y = F.vflip(y)
            if "target" in metadata and "mask" in metadata["target"]:
                metadata["target"]["mask"] = F.vflip(metadata["target"]["mask"])

        return x, y, metadata
