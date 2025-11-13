# Weather-4-Cast

This repository holds our official submission for the Weather-4-Cast 2025 competition. Our team competed in the 'Cumulative Rainfall Downstream' task.

Available at: https://weather4cast.net/neurips2025/

## Training configuration

| Parameter | Value |
| :--- | :--- |
| Batch Size | 32 |
| Gradient Accumulation Iterations | 128 |
| Effective Batch Size | 4096 |
| Start Learning Rate | 1.0e-05 |
| Learning Rate | 5.0e-04 |
| Final Learning Rate | 1.0e-05 |
| Weight Decay (wd) | 0.2 |
| Epochs | 30 |
| Rain Sampling Probability | 0.75 |
| Rain Sampling Threshold | 0.3 |


## Checkpoint

The URL for the best checkpoint is available [here](https://drive.google.com/file/d/1guhKowB4B2MlRleJ3NbcBzpHmrhSniD-/view?usp=sharing).

| Attribute | Value |
| :--- | :--- |
| Best epoch | 6 |
| Score | 3.510237 |
| Number of bins | 25601 |
| Bin maximum value (mm/hr) | 128.0 |