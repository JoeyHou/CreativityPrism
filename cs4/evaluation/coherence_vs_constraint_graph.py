import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import os
from quc_and_rcs import load_grouped_dfs_from_json
import copy

# Dictionary mapping model short names to full names
model_dict = {
    "gemma": "Gemma-7B Instruct",
    "mistral": "Mistral-7B Instruct",
    "llama": "Llama-2-7B Chat",
    "olmo_basehf": "OLMo Base",
    "olmo_sft": "OLMo SFT",
    "olmo_instruct": "OLMo Instruct"
}

# Function to annotate points on the plot
def annotate_points(x, y, labels, color):
    for i, label in enumerate(labels):
        annotation_text = f"Constraints: {label}"
        offset_x = random.randint(-15, 15)
        offset_y = random.randint(-15, 15)
        plt.annotate(
            annotation_text, (x[i], y[i]),
            textcoords="offset points",
            xytext=(offset_x, 5 + offset_y),
            ha='center',
            fontsize=12,
            fontstyle='italic',
            color=color
        )

# Function to process data and generate the plot
def process_and_plot_normalized(grouped_results, title_suffix, output_dir, save_as_pdf=False, pdf_filename="plot.pdf"):
    sns.set(style="whitegrid")
    # import seaborn as sns

    palette = sns.color_palette("husl", len(grouped_results))
    # Set colors based on available models
    colors = ['tab:blue', 'tab:orange', 'tab:green'] if 'gemma' in grouped_results else ['tab:red', 'tab:purple', 'tab:brown']

    plt.figure(figsize=(12, 7))
    idx = 0
    for model, grouped_model_df in grouped_results.items():
        plt.plot(
            grouped_model_df['average_percentage_gpt4'], 
            grouped_model_df['normalized_coherence_score'],
            label=model_dict.get(model, model), 
            marker='o', 
            # color=colors[idx%(len(colors))]
            # color=colors[colors]
            color=palette[idx]
        )
        annotate_points(
            grouped_model_df['average_percentage_gpt4'].values,             
            grouped_model_df['normalized_coherence_score'].values,             
            grouped_model_df.index, 
            palette[idx])
        idx += 1

    plt.xlabel('Constraint Satisfaction (%)', fontsize=14)
    plt.ylabel('Normalized Coherence Score', fontsize=14)
    plt.legend(title='Model', fontsize=11, title_fontsize='14')
    plt.grid(True)

    if save_as_pdf:
        pdf_path = os.path.join(output_dir, pdf_filename)
        plt.savefig(pdf_path, format='pdf')
        print(f"Plot saved as {pdf_path}")
    else:
        plt.show()

def merge_df(eval_constraints, eval_quality):
    merged_df = copy.deepcopy(eval_constraints)

    # get coherence score
    def get_coherence(row):
        if row['order'] == 1:
            return row['coherence_score_A']
        else:
            return row['coherence_score_B']
    coherence_score = eval_quality.apply(get_coherence, axis = 1)
    merged_df['coherence_score'] = coherence_score.values

    # get sati ratio
    merged_df['Percentage_GPT4'] = merged_df.apply(
        lambda row: row['satisfied'] / row['Number_of_Constraints'],
        axis = 1
    )
    return merged_df

# Main function to load data and generate the plot
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot Normalized Coherence Score by Constraint Satisfaction")
    
    # Arguments for input CSV files and output directory
    # parser.add_argument("--input_csv", nargs='+', required=True, help="List of CSV files for each model (format: model_name file_path).")
    # parser.add_argument("--output_dir", required=True, help="Directory to save the output plot.")
    parser.add_argument("--batch_id", required=True)
    parser.add_argument("--save_as_pdf", action="store_true", help="Flag to save the plot as PDF instead of displaying it.")
    
    args = parser.parse_args()

    # 1. prepare csv files
    batch_dir = './story_generator/output/{}/'.format(args.batch_id)
    model_lst = [d for d in os.listdir(batch_dir) if os.path.isdir(batch_dir + d)]
    exclude_lst = [
        "qwen_72b_instruct_qwen",
        "qwen_72b_instruct_llama",
        "qwen_72b_instruct_gpt",
        "deepseek_r1"
    ]

    model_lst = [
        "qwen_72b_instruct",
        "llama3_8b_instr",
        "llama3_70b_instruct",
        "mistral_7b_instr",
        "mistral_small_24b",
        "mixtral_8x7b",
        "olmo_7b",
        "olmo_13b",
    ]

    # print(model_lst)
    file_dict = {}
    for model in model_lst:
        if model in exclude_lst: continue 
        file_path = batch_dir + model + '/'
        try:
            eval_constraints = pd.read_csv(file_path + 'eval_constraints.csv')
            eval_quality = pd.read_csv(file_path + 'eval_quality.csv')
        except:
            continue 
        file_dict[model] = (eval_constraints, eval_quality)
    # print('=> Finished loading data! file_dict:', file_dict.keys())
    # print('=> model_lst:', model_lst, 'file_path:', file_path)
    # output_dir = batch_dir

    # 2. Load data from CSV files into grouped DataFrames
    grouped_results = {}
    for model, eval_results in file_dict.items():
        # df = pd.read_csv(file_path)
        # eval_constraints = pd.read_csv(file_path + 'eval_constraints.csv')
        # eval_quality = pd.read_csv(file_path + 'eval_quality.csv')
        eval_constraints, eval_quality = eval_results
        df = merge_df(eval_constraints, eval_quality)
        grouped_model_df = df.groupby('Number_of_Constraints').agg(
            total_coherence_score=('coherence_score', 'mean'),
            total_percentage_gpt4=('Percentage_GPT4', 'sum') # "Percentage_GPT4" means satisfaction rate
        )
        grouped_model_df['average_percentage_gpt4'] = grouped_model_df['total_percentage_gpt4'] / df['Number_of_Constraints'].value_counts()
        # max_coherence_score = grouped_model_df['total_coherence_score'].max()
        max_coherence_score = 5 # EDIT by Joey
        grouped_model_df['normalized_coherence_score'] = grouped_model_df['total_coherence_score'] / max_coherence_score
        grouped_results[model] = grouped_model_df

    # 3. Generate and save or display the plot
    process_and_plot_normalized(grouped_results, "Coherence vs Constraints", batch_dir, args.save_as_pdf, "coherence_vs_constraints.pdf")

if __name__ == "__main__":
    main()





# """# Quality Under Constraints

# QUC_n = (Normalized coherence score given n constraints) * (constraint satisfaction percentage given n constraints).
# """

# def calculate_quc_and_rcs(grouped_results):
#     quc_results = {}
#     rcs_results = {}

#     for model, df in grouped_results.items():
#         quc = df['normalized_coherence_score'] * df['average_percentage_gpt4']
#         quc_results[model] = quc

#         rcs = {}
#         constraints = df.index.tolist()
#         for i in range(len(constraints)):
#             for j in range(i + 1, len(constraints)):
#                 m = constraints[i]
#                 n = constraints[j]
#                 rcs[f"{m}-{n}"] = quc.loc[m] - quc.loc[n]

#         rcs_results[model] = rcs

#     return quc_results, rcs_results


# def Quc_9VsRcs_7_39(quc_results, rcs_results):
#     # Initialize an empty list to store the comparison data
#     comparison_data = []

#     for model in quc_results.keys():
#         # Extract QUC_39 for the model
#         quc_39 = quc_results[model].get("39", None)

#         # Extract RCS_7-39 for the model
#         rcs_7_39 = rcs_results[model].get('39-7', None)

#         # Append the data to the comparison list
#         comparison_data.append({
#             'Model': model,
#             'QUC_39': quc_39,
#             'RCS_7-39': rcs_7_39
#         })

#     # Create a DataFrame to display the comparison table
#     return pd.DataFrame(comparison_data)
