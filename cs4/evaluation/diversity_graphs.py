import pandas as pd
import matplotlib.pyplot as plt
import argparse

# The below code takes 3 CSV files as input. These files contain two columns: Number_of_Constraints, Product_diversity.
# The code aggregates the diversity scores for each constraint number and draws the diversity graphs.

def main(file1, file2, file3, output_path, label1, label2, label3):

    def prepare_data(df):
        # Function to calculate mean diversity grouped by 'Number_of_Constraints'
        return df.groupby('Number_of_Constraints')["Product_diversity"].mean()

    # Load the data from CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    # Preparing data for each dataframe
    data1 = prepare_data(df1)
    data2 = prepare_data(df2)
    data3 = prepare_data(df3)

    # Plotting all three datasets on the same graph - modify the labels as required.
    plt.figure(figsize=(12, 6))
    data1.plot(kind='line', marker='o', label=label1)
    data3.plot(kind='line', marker='o', label=label2)
    data2.plot(kind='line', marker='o', label=label3)
    constraints = [7, 15, 23, 31, 39]

    plt.xlabel('Number of Constraints', fontsize=14)
    plt.xticks(constraints)
    plt.legend(title="Model", fontsize=12, title_fontsize='12')

    plt.ylabel('Average Diversity', fontsize=14)
    plt.grid(True)

    # Save the plot to the specified output file
    plt.savefig(output_path)

    print(f"\nDiversity Graph for {label1}, {label2} and {label3} saved in provided location!\n" )

if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(description="Aggregate diversity scores and plot the diversity graphs.")
    
    # Adding arguments for input CSV file paths and output file path
    parser.add_argument('--file1', required=True, help="Path to the first CSV file")
    parser.add_argument('--file2', required=True, help="Path to the second CSV file")
    parser.add_argument('--file3', required=True, help="Path to the third CSV file")
    parser.add_argument('--label1', default="Gemma-7B Instruct", help="Label for the first model (default: Gemma-7B Instruct)")
    parser.add_argument('--label2', default="Llama-2-7B Chat", help="Label for the second model (default: Llama-2-7B Chat)")
    parser.add_argument('--label3', default="Mistral-7B Instruct", help="Label for the third model (default: Mistral-7B Instruct)")
    parser.add_argument('--output_path', required=True, help="Path to save the output plot image")

    # Parsing the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.file1, args.file2, args.file3, args.output_path, args.label1, args.label2, args.label3)
