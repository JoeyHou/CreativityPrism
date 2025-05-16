import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_average_perplexity(file1, file2, file3, label1, label2, label3, output_path):
    # Load 3 CSV files - these files have two columns (Number_of_Constraints, Perplexity)
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    # Preparing data for each dataframe - group perplexity values by the number of constraints.
    data1 = df1.groupby('Number_of_Constraints')['Perplexity'].mean()
    data2 = df2.groupby('Number_of_Constraints')['Perplexity'].mean()
    data3 = df3.groupby('Number_of_Constraints')['Perplexity'].mean()

    # Plotting all three datasets on the same graph
    plt.figure(figsize=(12, 6))
    data1.plot(kind='line', marker='o', label=label1)
    data2.plot(kind='line', marker='^', label=label2)
    data3.plot(kind='line', marker='s', label=label3)
    plt.title('Average Perplexity by Number of Constraints')
    plt.xlabel('Number of Constraints')
    plt.ylabel('Average Perplexity')
    plt.grid(True)
    plt.legend()

    # Save the plot to a file
    plt.savefig(output_path)

    print(f"\nPerplexity Graph for {label1}, {label2} and {label3} saved in provided location!\n" )


if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(description="Plot average perplexity by number of constraints.")
    
    # Adding arguments for file paths and labels
    parser.add_argument('--file1', required=True, help="Path to the first CSV file")
    parser.add_argument('--file2', required=True, help="Path to the second CSV file")
    parser.add_argument('--file3', required=True, help="Path to the third CSV file")
    parser.add_argument('--label1', default="Gemma-7B Instruct", help="Label for the first model (default: Gemma-7B Instruct)")
    parser.add_argument('--label2', default="Llama-2-7B Chat", help="Label for the second model (default: Llama-2-7B Chat)")
    parser.add_argument('--label3', default="Mistral-7B Instruct", help="Label for the third model (default: Mistral-7B Instruct)")
    parser.add_argument('--output_path', required=True, help="Path to save the output plot image")

    # Parsing the arguments
    args = parser.parse_args()

    # Call the function with arguments
    plot_average_perplexity(args.file1, args.file2, args.file3, args.label1, args.label2, args.label3, args.output_path)
