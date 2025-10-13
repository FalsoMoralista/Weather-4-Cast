import torch
import pandas as pd
import os


def generate_submission_files(predictions_dir, dictionary_dir, output_dir):
    """
    Generates submission files based on model predictions and dictionary files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    task = "cum1"
    years = ["19", "20"]
    filenames = ["roxi_0008", "roxi_0009", "roxi_0010"]

    for year in years:
        predictions_path = os.path.join(
            predictions_dir, f"predictions_vjepa2_{task}test{year}.pth"
        )
        try:
            predictions_data = torch.load(predictions_path)
            print(f"\nLoaded predictions from: {predictions_path}")
        except FileNotFoundError:
            print(f"Warning: Predictions file not found for year {year}. Skipping.")
            continue

        tensor_start_index = 0

        for name in filenames:
            prediction_key = f"{name}.{task}test{year}"

            if prediction_key not in predictions_data:
                print(
                    f"Warning: Key '{prediction_key}' not in predictions file. Skipping."
                )
                continue

            full_predictions_tensor = predictions_data[prediction_key]
            print(
                f"Predictions shape for key: {prediction_key}",
                full_predictions_tensor.shape,
            )

            dict_path = os.path.join(
                dictionary_dir, f"{name}.{task}test_dictionary.csv"
            )
            try:
                meta_df = pd.read_csv(dict_path)
            except FileNotFoundError:
                print(
                    f"Warning: Dictionary file not found for {name} and year {year}. Skipping."
                )
                continue

            year_full = int(f"20{year}")
            year_specific_df = meta_df[meta_df["year"] == year_full].reset_index(
                drop=True
            )

            num_samples_for_file = len(year_specific_df)
            if num_samples_for_file == 0:
                print(f"No data for {name} in year {year_full}. Skipping.")
                continue

            file_specific_tensor = full_predictions_tensor[
                tensor_start_index : tensor_start_index + num_samples_for_file
            ]
            print(
                f"Processing {name} for year {year_full}: {num_samples_for_file} samples."
            )

            tensor_start_index += num_samples_for_file

            submission_results = []

            for i, row in year_specific_df.iterrows():
                # Get the single sample tensor for this case
                # The original code had a bug here, it should slice `file_specific_tensor`
                slot_start, slot_end = row["slot-start"], row["slot-end"]

                sample_tensor = full_predictions_tensor[
                    slot_start:slot_end
                ]  # Shape: (16, 252, 252)

                # --- START: REVISED LOGIC FOR ACCURATE AVERAGING ---

                # 1. High-resolution (HR) coordinates of the 32x32 ROI
                hr_y_start, hr_y_end = row["y-top-left"], row["y-bottom-right"]
                hr_x_start, hr_x_end = row["x-top-left"], row["x-bottom-right"]

                # 2. Find the corresponding range of low-resolution (LR) pixels
                # These are the pixels we need to consider from the 252x252 grid
                lr_y_start_idx = hr_y_start // 6
                lr_y_end_idx = (hr_y_end - 1) // 6
                lr_x_start_idx = hr_x_start // 6
                lr_x_end_idx = (hr_x_end - 1) // 6

                # 3. Extract the relevant patch from the prediction tensor
                prediction_patch = sample_tensor[
                    :,
                    :,
                    lr_y_start_idx : lr_y_end_idx + 1,
                    lr_x_start_idx : lr_x_end_idx + 1,
                ]

                print("Prediction patch shape:", prediction_patch.shape)

                dim = (2, 3)

                first_hour_mean = torch.mean(prediction_patch[:, :4, :], dim=dim)
                print("First hour mean shape:", first_hour_mean.shape)
                first_hour_mean = torch.mean(first_hour_mean, dim=1)
                print("First hour mean shape:", first_hour_mean.shape)

                second_hour_mean = torch.mean(prediction_patch[:, 4:8, :], dim=dim)
                print("Second hour mean shape:", second_hour_mean.shape)
                second_hour_mean = torch.mean(second_hour_mean, dim=1)
                print("Second hour mean shape:", second_hour_mean.shape)

                third_hour_mean = torch.mean(prediction_patch[:, 8:12, :], dim=dim)
                print("Third hour mean shape:", third_hour_mean.shape)
                third_hour_mean = torch.mean(third_hour_mean, dim=1)
                print("Third hour mean shape:", third_hour_mean.shape)

                fourth_hour_mean = torch.mean(prediction_patch[:, 12:16, :], dim=dim)
                print("Fourth hour mean shape:", fourth_hour_mean.shape)
                fourth_hour_mean = torch.mean(fourth_hour_mean, dim=1)
                print("Fourth hour mean shape:", fourth_hour_mean.shape)

                rain_mean = torch.stack(
                    [
                        first_hour_mean,
                        second_hour_mean,
                        third_hour_mean,
                        fourth_hour_mean,
                    ]
                )
                print("Rain mean stack shape:", rain_mean.shape)

                total_rain = torch.sum(rain_mean)
                print("Total rain shape", total_rain.shape)
                total_rain = total_rain.item()
                print("Total rain:", total_rain)

                # mean = torch.mean(prediction_patch, dim=(1, 2))
                # print("Mean shape:", mean.shape)

                # total_rain = torch.sum(mean).item()

                # # 4. Create a weight matrix to store the overlap area for each LR pixel
                # patch_h, patch_w = prediction_patch.shape[2], prediction_patch.shape[3]
                # weight_matrix = torch.zeros(
                #     (patch_h, patch_w), device=prediction_patch.device
                # )

                # for y_idx in range(patch_h):
                #     for x_idx in range(patch_w):
                #         # Global LR coordinates
                #         global_lr_y = lr_y_start_idx + y_idx
                #         global_lr_x = lr_x_start_idx + x_idx

                #         # Corresponding HR coordinates for this LR pixel's 6x6 block
                #         pixel_hr_y_start = global_lr_y * 6
                #         pixel_hr_y_end = pixel_hr_y_start + 6
                #         pixel_hr_x_start = global_lr_x * 6
                #         pixel_hr_x_end = pixel_hr_x_start + 6

                #         # Calculate the intersection area with the 32x32 ROI
                #         overlap_y = max(
                #             0,
                #             min(hr_y_end, pixel_hr_y_end)
                #             - max(hr_y_start, pixel_hr_y_start),
                #         )
                #         overlap_x = max(
                #             0,
                #             min(hr_x_end, pixel_hr_x_end)
                #             - max(hr_x_start, pixel_hr_x_start),
                #         )

                #         weight_matrix[y_idx, x_idx] = overlap_y * overlap_x

                # # 5. Calculate the weighted sum and normalize
                # # The total area of the ROI is (hr_y_end - hr_y_start) * (hr_x_end - hr_x_start), which is 32*32=1024
                # total_roi_area = (hr_y_end - hr_y_start) * (hr_x_end - hr_x_start)

                # # Multiply the prediction patch by the weights and sum everything up
                # # weight_matrix is (H, W), prediction_patch is (16, H, W). Broadcasting handles this.
                # weighted_sum = torch.sum(prediction_patch * weight_matrix)

                # # The final value is the weighted average. We divide by the total area of all frames.
                # prediction_value = (weighted_sum / (total_roi_area * 16)).item()

                # --- END: REVISED LOGIC ---

                submission_results.append([row["Case-id"], total_rain, 1])

            output_df = pd.DataFrame(submission_results, columns=None)
            output_filename = os.path.join(
                output_dir, f"20{year}/{name}.test.cum4h.csv"
            )
            output_df.to_csv(output_filename, index=False, header=False)
            print(f"Successfully generated submission file: {output_filename}")


# --- How to Use ---
PREDICTIONS_DIR = "predictions"
DICTIONARY_DIR = "submissions"
OUTPUT_DIR = "submission_files"

generate_submission_files(PREDICTIONS_DIR, DICTIONARY_DIR, OUTPUT_DIR)
