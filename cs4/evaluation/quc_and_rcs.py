import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

# Function to calculate QUC and RCS
def calculate_quc_and_rcs(grouped_results):
    quc_results = {}
    rcs_results = {}

    for model, df in grouped_results.items():
        quc = df['normalized_coherence_score'] * df['average_percentage_gpt4']
        quc_results[model] = quc

        rcs = {}
        constraints = df.index.tolist()
        for i in range(len(constraints)):
            for j in range(i + 1, len(constraints)):
                m = constraints[i]
                n = constraints[j]
                rcs[f"{n}-{m}"] = quc.loc[n] - quc.loc[m]

        rcs_results[model] = rcs

    return quc_results, rcs_results

# Function to load grouped DataFrames from a JSON file
def load_grouped_dfs_from_json(filename):
    with open(filename, 'r') as file:
        json_dict = json.load(file)
    
    grouped_dfs = {key: pd.DataFrame(value) for key, value in json_dict.items()}
    return grouped_dfs

# Function to visualize QUC vs Number of Constraints for two types of constraints
def plot_quc(type1_quc, type2_quc, model_dict, output_dir):
    sns.set(style="whitegrid")

    # Plotting Type 1 constraints
    plt.figure(figsize=(10, 6))
    for model, series_data in type1_quc.items():
        plt.plot(series_data.index, series_data.values, marker='o', label=model_dict[model])
    plt.xlabel('Number of Constraints', fontsize=14)
    plt.ylabel('QUC', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.gca().invert_xaxis()  # Reverse the x-axis
    plt.savefig(os.path.join(output_dir, "type1_quc.pdf"), format='pdf')
    plt.close()

    # Plotting Type 2 constraints
    plt.figure(figsize=(10, 6))
    for model, series_data in type2_quc.items():
        plt.plot(series_data.index, series_data.values, marker='o', label=model_dict[model])
    plt.xlabel('Number of Constraints', fontsize=14)
    plt.ylabel('QUC', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.gca().invert_xaxis()  # Reverse the x-axis
    plt.savefig(os.path.join(output_dir, "type2_quc.pdf"), format='pdf')
    plt.close()

# Main function to handle argument parsing
def main():
    parser = argparse.ArgumentParser(description="Calculate and plot QUC and RCS from input data.")
    
    # Input and output paths
    parser.add_argument("--input_json", required=True, help="Path to input JSON file containing grouped results.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output plots.")
    
    args = parser.parse_args()

    # Load the grouped DataFrames from JSON file
    grouped_dfs = load_grouped_dfs_from_json(args.input_json)
    
    # Calculate QUC and RCS
    quc_results, rcs_results = calculate_quc_and_rcs(grouped_dfs)

    # Define model names
    model_dict = {
        "gemma": "Gemma-7B Instruct",
        "mistral": "Mistral-7B Instruct",
        "llama": "Llama-2-7B Chat",
        "olmo_basehf": "OLMo Base",
        "olmo_sft": "OLMo SFT",
        "olmo_instruct": "OLMo Instruct"
    }

    # Separate data for two types of constraints
    type1_quc = {key: quc_results[key] for key in ['d2_mgl_quc', 'd2_olmo_quc']}
    type2_quc = {key: quc_results[key] for key in ['d3_mgl_quc', 'd3_olmo_quc']}

    # Plot and save the figures
    plot_quc(type1_quc, type2_quc, model_dict, args.output_dir)

if __name__ == "__main__":
    main()
