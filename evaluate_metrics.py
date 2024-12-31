import yaml
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List
import ast

# Function to read all YAML files and load them into a DataFrame
def load_yaml_data(yaml_folder):
    data = []
    
    for filename in os.listdir(yaml_folder):
        if filename.endswith(".yaml"):
            filepath = os.path.join(yaml_folder, filename)
            with open(filepath, 'r') as file:
                content = yaml.safe_load(file)
                for key, value in content.items():
                    # Add an identifier for the experiment
                    value['experiment_id'] = key
                    data.append(value)
    
    df = pd.json_normalise(data, sep='_')
    return df

def normalise_tp_fp_fn_tn(df):
    """
    Normalises the values of TP, FP, FN, and TN with respect to the total of P and N.
    P is the total positives (TP + FN), and N is the total negatives (TN + FP).

    Parameters:
    df (pd.DataFrame): The DataFrame containing the TP, FP, FN, and TN columns.

    Returns:
    pd.DataFrame: The DataFrame with the normalised values.
    """
    # Check that the required columns are present in the DataFrame
    required_columns = ['metrics_train/TP', 'metrics_train/FP', 'metrics_train/FN', 'metrics_train/TN']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The DataFrame must contain the columns {required_columns}")

    # Calculate the total of P and N
    df['P'] = df['metrics_train/TP'] + df['metrics_train/FN']
    df['N'] = df['metrics_train/TN'] + df['metrics_train/FP']

    # Normalise TP, FP, FN, and TN
    df['TP_normalised'] = df['metrics_train/TP'] / df['P']
    df['FP_normalised'] = df['metrics_train/FP'] / df['P']
    df['FN_normalised'] = df['metrics_train/FN'] / df['P']
    df['TN_normalised'] = 0

    # Remove intermediate columns
    df.drop(columns=['P', 'N'], inplace=True)

    return df

# Function to list characteristics ordered by recall
def list_by_recall(df):
    sorted_df = df.sort_values(by='metrics_train/Recall', ascending=False)
    return sorted_df[['characteristics', 'metrics_train/Recall']]

# Function to list characteristics ordered by precision
def list_by_precision(df):
    sorted_df = df.sort_values(by='metrics_train/Precision', ascending=False)
    return sorted_df[['characteristics', 'metrics_train/Precision']]

def list_by(df: pd.DataFrame, metric: str, metrics_to_show: List[str]):
    sorted_df = df.sort_values(by=metric, ascending=False)
    return sorted_df[['characteristics'] + [metric] + metrics_to_show]

def save_dataframe_to_excel(df, store_path, columns: List[str], filename: str = 'evaluation_summary.csv'):
    """
    Saves a DataFrame to an Excel file.

    Parameters:
    df (pd.DataFrame): The DataFrame to be saved.
    columns (List[str]): Columns to be stored.
    filename (str): The name of the output Excel file (must end with '.csv').
    """
    df = df[columns]
    
    df.to_csv(os.path.join(store_path, filename), index=False, sep=';', decimal=',')
    print(f"DataFrame saved to {filename}")

# Function to plot the curves
def plot_curves(df):

    # Create a figure with 2x2 subplots for the four graphs (multi-plot)
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # Create a list to store legend labels and ensure they are not duplicated
    legends = []

    # 1. Recall vs Precision
    for _, row in df.iterrows():
        characteristics = ast.literal_eval(row['characteristics'])
        characteristics_labels = ', '.join([str(c) for c in characteristics])  # Join characteristics into a string
        
        if 'metrics_train/curves_RP_recall' in row and 'metrics_train/curves_RP_precision' in row:
            recall = row['metrics_train/curves_RP_recall']
            precision = row['metrics_train/curves_RP_precision'][0]
            if recall and precision:
                axs[0, 0].plot(recall, precision)
                legends.append(characteristics_labels)
    
    axs[0, 0].set_xlabel('Recall')
    axs[0, 0].set_ylabel('Precision')
    axs[0, 0].grid(True)

    # 2. Confidence vs Precision
    for _, row in df.iterrows():
        characteristics = ast.literal_eval(row['characteristics'])
        characteristics_labels = ', '.join([str(c) for c in characteristics])
        
        if 'metrics_train/curves_CP_confidence' in row and 'metrics_train/curves_CP_precision' in row:
            confidence_cp = row['metrics_train/curves_CP_confidence']
            precision_cp = row['metrics_train/curves_CP_precision'][0]
            if confidence_cp and precision_cp:
                axs[0, 1].plot(confidence_cp, precision_cp)
    
    axs[0, 1].set_xlabel('Confidence')
    axs[0, 1].set_ylabel('Precision')
    axs[0, 1].grid(True)

    # 3. Confidence vs Recall
    for _, row in df.iterrows():
        characteristics = ast.literal_eval(row['characteristics'])
        characteristics_labels = ', '.join([str(c) for c in characteristics])
        
        if 'metrics_train/curves_CR_confidence' in row and 'metrics_train/curves_CR_recall' in row:
            confidence_cr = row['metrics_train/curves_CR_confidence']
            recall_cr = row['metrics_train/curves_CR_recall'][0]
            if confidence_cr and recall_cr:
                axs[1, 0].plot(confidence_cr, recall_cr)
    
    axs[1, 0].set_xlabel('Confidence')
    axs[1, 0].set_ylabel('Recall')
    axs[1, 0].grid(True)

    # 4. Confidence vs F1 Score
    for _, row in df.iterrows():
        characteristics = ast.literal_eval(row['characteristics'])
        characteristics_labels = ', '.join([str(c) for c in characteristics])
        
        if 'metrics_train/curves_CF_confidence' in row and 'metrics_train/curves_CF_F1' in row:
            confidence_cf = row['metrics_train/curves_CF_confidence']
            f1_score_cf = row['metrics_train/curves_CF_F1'][0]
            if confidence_cf and f1_score_cf:
                axs[1, 1].plot(confidence_cf, f1_score_cf)
    
    axs[1, 1].set_xlabel('Confidence')
    axs[1, 1].set_ylabel('F1 Score')
    axs[1, 1].grid(True)

    # Adjust spacing between subplots to avoid overlap
    plt.tight_layout()

    # Add a legend outside the plots (common to all)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(legends, loc='upper centre', ncol=3, fontsize='small')

    # Show all plots together
    plt.show()

# Main function
def main():
    # Path to the folder of YAML files
    yaml_folder = r'C:...\yamls'

    target_metric = 'metrics_train/Recall'
    other_interesting_metrics = ['metrics_train/Precision', 'metrics_train/F1Score', 'metrics_train/speed_inference']
    
    # Load the data
    df = load_yaml_data(yaml_folder)
    
    df = normalise_tp_fp_fn_tn(df)
    # List characteristics ordered by recall and precision
    print("Characteristics ordered by Recall (highest to lowest):")
    print(list_by_recall(df))
    
    print("\nCharacteristics ordered by Precision (highest to lowest):")
    print(list_by_precision(df))

    target_metric = 'metrics_train/Recall'
    metrics = ['params_dataset', 'params_model', 'metrics_train/mAP_0.5', 'metrics_train/mAP_0.5:0.95','metrics_train/Precision', 'metrics_train/F1Score', 'metrics_train/speed_inference', 'TP_normalised', 'FP_normalised', 'TN_normalised', 'FN_normalised','metrics_train/TN', 'metrics_train/TP','metrics_train/FN', 'metrics_train/FP']
    df_short = list_by(df, target_metric, metrics)
    print(df_short)
    
    save_dataframe_to_excel(df_short, yaml_folder, ['characteristics'] + [target_metric] + metrics)
    # Plot the curves
    plot_curves(df)

# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()
