import torch
import pandas as pd
import os
import math

def generate_submission_files(predictions_dir, dictionary_dir, output_dir):
    """
    Generates submission files based on model predictions and dictionary files.

    Args:
        predictions_dir (str): Directory containing the .pth prediction files.
        dictionary_dir (str): Directory containing the .cum1test_dictionary.csv files.
        output_dir (str): Directory to save the final submission .csv files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Configuration from your run_inference.py script
    task = 'cum1'
    years = ['19', '20']
    filenames = ['roxi_0008', 'roxi_0009', 'roxi_0010']

    for year in years:
        # Load the predictions file for the current year
        predictions_path = os.path.join(predictions_dir, f'predictions_{task}test{year}.pth')
        try:
            predictions_data = torch.load(predictions_path)
            print(f"\nLoaded predictions from: {predictions_path}")
        except FileNotFoundError:
            print(f"Warning: Predictions file not found for year {year}. Skipping.")
            continue

        # The prediction tensor contains data for all filenames concatenated
        # We need to track our position in this large tensor
        tensor_start_index = 0

        for name in filenames:
            # The key in the dictionary corresponds to the file type
            prediction_key = f'{name}.{task}test{year}'
            
            if prediction_key not in predictions_data:
                print(f"Warning: Key '{prediction_key}' not in predictions file. Skipping.")
                continue

            # This is the full tensor for the year, e.g., (97, 16, 252, 252) for year '19'
            full_predictions_tensor = predictions_data[prediction_key]

            # Load the corresponding dictionary file
            dict_path = os.path.join(dictionary_dir, f'{name}.{task}test_dictionary.csv')
            try:
                meta_df = pd.read_csv(dict_path)
            except FileNotFoundError:
                print(f"Warning: Dictionary file not found for {name} and year {year}. Skipping.")
                continue
            
            # Filter the metadata for the current year (e.g., 2019 from the 'year' column)
            year_full = int(f"20{year}")
            year_specific_df = meta_df[meta_df['year'] == year_full].reset_index(drop=True)
            
            num_samples_for_file = len(year_specific_df)
            if num_samples_for_file == 0:
                print(f"No data for {name} in year {year_full}. Skipping.")
                continue

            # Get the slice of the tensor corresponding to the current file
            tensor_end_index = tensor_start_index + num_samples_for_file
            file_specific_tensor = full_predictions_tensor[tensor_start_index:tensor_end_index]
            print(f"Processing {name} for year {year_full}: {num_samples_for_file} samples.")
            
            # This becomes the start index for the next file in the loop
            tensor_start_index = tensor_end_index

            submission_results = []

            
            for i, row in year_specific_df.iterrows():
                case_id = row['Case-id']

                
                x_top_left, x_bottom_right = row['x-top-left']//6, row['x-bottom-right']//6
                y_top_left, y_bottom_right = row['y-top-left']//6, row['y-bottom-right']//6

                slot_start, slot_end = row['slot-start'], row['slot-end']

                # Get the prediction for this specific case (sample)
                # Shape: (16, 252, 252)
                sample_tensor = full_predictions_tensor[slot_start:slot_end]
                print('Sample Tensor:', sample_tensor.size(),flush=True)
                # Extract the central 32x32 region
                # Shape: (16, 32, 32)
                region_of_interest = sample_tensor[:, :, y_top_left:y_bottom_right, x_top_left:x_bottom_right]
                print('region_of_interest:', region_of_interest.size(),flush=True)
                # Calculate the mean across all 16 frames and the 32x32 pixels
                prediction_value = torch.mean(region_of_interest).item()
                print('prediction value:', prediction_value, flush=True)
                # The submission format requires an 'Hour' column, which is '1' in your example
                submission_results.append({
                    'Case-id': case_id,
                    'Prediction': prediction_value,
                    'Hour': 1
                })

            # Create and save the submission DataFrame
            output_df = pd.DataFrame(submission_results)
            output_filename = os.path.join(output_dir, f'{name}.test.cum4h_{year}.csv')
            output_df.to_csv(output_filename, index=False)
            print(f"Successfully generated submission file: {output_filename}")


# --- How to Use ---
# 1. Place your prediction .pth files in a 'predictions' directory.
# 2. Place your dictionary .csv files in a 'dictionaries' directory.
# 3. An 'output' directory will be created for the results.

# Assuming the script is in the same folder as these directories
PREDICTIONS_DIR = 'predictions' # Directory with your .pth files
DICTIONARY_DIR = 'submissions'  # Directory with your roxi_..._dictionary.csv files
OUTPUT_DIR = 'submission_files'      # Where to save the final .csv files

generate_submission_files(PREDICTIONS_DIR, DICTIONARY_DIR, OUTPUT_DIR)
