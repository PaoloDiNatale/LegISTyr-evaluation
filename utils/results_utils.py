import pandas as pd
import os

from .term_finder_utils import *

def find_terms_over_models(nlp, entries_dict, models_list, domain):
    """
    Find terms across all translation models using the specified domain.
    
    Parameters:
    - nlp: spaCy NLP model
    - entries_dict (dict): Dictionary mapping model names to entry lists
    - models_list (list): List of model/column names to process
    - domain (str): Domain to search in ("South-Tyrol", "other_tyrol", "other_systems", "homonym")
    
    Returns:
    - tuple: (term_finders dict, term_results dict)
        - term_finders: TermFinder instances for each model
        - term_results: Matched terms for each model
    """
    
    term_finders = {}
    term_results = {}
    
    for col in models_list:
        entry_list = entries_dict[col]
        
        tf = TermFinder(nlp, entry_list)
        # Store the instance if you want to reuse it
        term_finders[col] = tf
        
        # Find terms matching the specified domain
        print(f"Matching {domain} terms for model: {col}")
        term_match = tf.find_terms(domain=domain)
        term_results[col] = term_match
    
    return term_results



def save_term_results(term_results, filename, output_dir="./data/results"):
    """
    Convert term results dictionary to a combined DataFrame and save to CSV.
    
    Parameters:
    - term_results (dict): Dictionary with column names as keys and 
                          {sentence: list_of_matches} dicts as values
    - output_dir (str): Directory where the CSV will be saved (default: "data")
    
    Returns:
    - pd.DataFrame: The combined DataFrame that was saved
    """
    
    term_results_dfs = {}
    
    # Convert each result to a DataFrame
    for col, result in term_results.items():
        # result is a dict: {sentence: list_of_matches}
        data = []
        
        for sentence, matches in result.items():
            # Extract matched text spans (as strings)
            match_texts = [span.text for span in matches]
            data.append({"matches": match_texts})
        
        # Create a DataFrame for this column
        term_results_dfs[col] = pd.DataFrame(data)
    
    # Initialize with the sentences from the first column
    base_col = list(term_results_dfs.keys())[0]
    combined_df = term_results_dfs[base_col][["matches"]].copy()
    combined_df.rename(columns={"matches": base_col}, inplace=True)
    
    # Add each match column under its corresponding translation name
    for col_name, df_match in term_results_dfs.items():
        if col_name != base_col:
            combined_df[col_name] = df_match["matches"].values
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.csv")
    combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Results saved to: {output_path}")
    return combined_df
    

def calculate_success_rate(data_dict):
    """
    Calculate the percentage of full (non-empty) lists in a dictionary.

    Args:
        data_dict (dict): A dictionary where values are lists.

    Returns:
        float: Percentage of full lists against total lists.
    """
    total_lists = len(data_dict)
    full_lists = sum(1 for v in data_dict.values() if isinstance(v, list) and v)

    if total_lists == 0:
        return 0.0

    return (full_lists / total_lists) * 100


def print_success_rate(term_results, category_name, output_dir="./data/results_analysis", filename="term_accuracy_rates", clear_file=False):
    """
    Calculate and save the percentage of non-empty lists ('matches') for each column.

    Args:
        term_results_dfs (dict): Dict of DataFrames, each with a 'matches' column.
        output_dir (str): Directory to save the summary CSV.
        filename (str): Name of the output CSV (without extension).
    """

    # Calculate success percentages for each dataframe
    full_percentage_results = {}

    for model_name, result_dict in term_results.items():
        # Here, result_dict is a dict: {sentence: list_of_matches}
        percentage = calculate_success_rate(result_dict)
        full_percentage_results[model_name] = round(percentage, 2)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    overwrite = "w" if clear_file else "a"

    # Append to text file
    with open(output_path, overwrite, encoding="utf-8") as f:
        f.write(f"\n{category_name}:\n")
        for model, rate in full_percentage_results.items():
            f.write(f"{model}: {rate:.2f}%\n")

    return full_percentage_results