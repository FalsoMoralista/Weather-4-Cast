import torch
import pandas as pd
import os
from sys import argv


def generate_submission_files(predictions_dir, dictionary_dir, output_dir, model):
    """
    Generates submission files based on model predictions and dictionary files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    if not os.path.exists(os.path.join(output_dir, model)):
        os.makedirs(os.path.join(output_dir, model))
        print(
            f"Created model-specific output directory: {os.path.join(output_dir, model)}"
        )

    if not os.path.exists(os.path.join(output_dir, model, "2019")):
        os.makedirs(os.path.join(output_dir, model, "2019"))
        print(
            f"Created year-specific output directory: {os.path.join(output_dir, model, '2019')}"
        )

    if not os.path.exists(os.path.join(output_dir, model, "2020")):
        os.makedirs(os.path.join(output_dir, model, "2020"))
        print(
            f"Created year-specific output directory: {os.path.join(output_dir, model, '2020')}"
        )

    task = "cum1"
    years = ["19", "20"]
    filenames = ["roxi_0008", "roxi_0009", "roxi_0010"]

    for year in years:
        predictions_path = os.path.join(
            predictions_dir, f"predictions_{model}_{task}test{year}.pth"
        )
        try:
            predictions_data = torch.load(predictions_path, map_location="cpu")
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

            print(
                f"Processing {name} for year {year_full}: {num_samples_for_file} samples."
            )

            tensor_start_index += num_samples_for_file

            submission_results = []

            for i, row in year_specific_df.iterrows():
                # Get the single sample tensor for this case
                # The original code had a bug here, it should slice `file_specific_tensor`
                slot_start, _ = row["slot-start"], row["slot-end"]
                slot = slot_start // 4

                print(f"Processing Case-id {row['Case-id']} with slot {slot}")

                sample_tensor = full_predictions_tensor[slot]
                print(
                    f"Sample tensor shape for Case-id {row['Case-id']}: {sample_tensor.shape}"
                )

                # --- START: REVISED LOGIC FOR ACCURATE AVERAGING ---

                # 1. High-resolution (HR) coordinates of the 32x32 ROI
                hr_y_start, hr_y_end = row["y-top-left"], row["y-bottom-right"]
                hr_x_start, hr_x_end = row["x-top-left"], row["x-bottom-right"]

                # 2. Find the corresponding range of low-resolution (LR) pixels
                # These are the pixels we need to consider from the 252x252 grid
                lr_y_start_idx = hr_y_start // 6
                lr_y_end_idx = hr_y_end // 6
                lr_x_start_idx = hr_x_start // 6
                lr_x_end_idx = hr_x_end // 6

                H, W = sample_tensor.shape[1], sample_tensor.shape[2]

                y1, y2 = max(0, lr_y_start_idx), min(H, lr_y_end_idx)
                x1, x2 = max(0, lr_x_start_idx), min(W, lr_x_end_idx)

                # 3. Extract the relevant patch from the prediction tensor
                prediction_patch = sample_tensor[:, y1:y2, x1:x2]

                print("Prediction patch shape:", prediction_patch.shape)

                def predict_simple(pred_tensor):
                    mean = torch.mean(pred_tensor)
                    return mean.item() * 4

                total_rain = predict_simple(prediction_patch)
                total_rain = max(0, total_rain)  # Ensure non-negative

                submission_results.append([row["Case-id"], 5.0, 1])

            output_df = pd.DataFrame(submission_results, columns=None)
            output_filename = os.path.join(
                output_dir, f"{MODEL}/20{year}/{name}.test.cum4h.csv"
            )
            output_df.to_csv(output_filename, index=False, header=False)
            print(f"Successfully generated submission file: {output_filename}")


# --- How to Use ---
PREDICTIONS_DIR = "predictions"
DICTIONARY_DIR = "submissions"
OUTPUT_DIR = "submission_files"
MODEL = argv[1] if len(argv) > 1 else "vanilla_jepa"

generate_submission_files(PREDICTIONS_DIR, DICTIONARY_DIR, OUTPUT_DIR, MODEL)
