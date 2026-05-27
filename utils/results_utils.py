# results_utils.py
import pandas as pd
import os

from .term_finder_utils import TermFinder

def find_terms_over_models(nlp, entries_dict, raw_entries_dict, models_list, domain, lang="de", tgt_term_columns=None):
    """
    Find terms across all translation models using the specified domain.
    
    Parameters:
    - nlp: spaCy NLP model
    - entries_dict (dict): Dictionary mapping model names to entry lists
    - raw_entries_dict (dict): Dictionary mapping model names to raw entry lists
    - models_list (list): List of model/column names to process
    - domain (str): Domain to search in
        - LegISTyr: "South-Tyrol", "other_tyrol", "other_systems", "homonym"
        - BISTRO: "tgt_term_1", "tgt_term_2", etc.
    - lang (str): Language code ('de' or 'it')
    - tgt_term_columns (list): For BISTRO - ordered list of tgt_term column names
    
    Returns:
    - dict: term_results mapping model names to matched terms
    """
    
    term_finders = {}
    term_results = {}
    
    for col in models_list:
        entry_list = entries_dict[col]
        raw_entry_list = raw_entries_dict[col]
        
        tf = TermFinder(nlp, entry_list, raw_entry_list, lang=lang, tgt_term_columns=tgt_term_columns)

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
 
    Rows where matches is None were skipped (no term annotation in source data)
    and are written as empty cells so the CSV index stays aligned with the input.
    
    Parameters:
    - term_results (dict): Dictionary with column names as keys and 
                          {sentence: list_of_matches | None} dicts as values
    - filename (str): Base filename (without extension)
    - output_dir (str): Directory where the CSV will be saved
    
    Returns:
    - pd.DataFrame: The combined DataFrame that was saved
    """
    
    term_results_dfs = {}
    
    # Convert each result to a DataFrame
    for col, result in term_results.items():
        # result is a dict: {sentence: list_of_matches | None}
        # None  → row had no term annotation; matcher was skipped
        # []    → matcher ran but found no matches
        # [...]  → matcher ran and found matches
        data = []
        
        for sentence, matches in result.items():
            if matches is None:
                # Skipped row: represent as NaN in the CSV so it is visually
                # distinct from an empty-match row (which saves as "[]")
                data.append({"matches": None})
            else:
                match_texts = [span.text for span in matches]
                data.append({"matches": match_texts})
        
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
    Calculate the success rate only over rows where the matcher was actually run.
 
    Rows stored as None were skipped because the source data had no term
    annotation (NaN). They are excluded from both numerator and denominator so
    they do not artificially deflate the reported success rate.
 
    Args:
        data_dict (dict): Mapping of sentence-id → list_of_matches | None.
 
    Returns:
        tuple[float, int, int]:
            percentage    – success rate over matcher-run rows (0.0 if none run)
            matched_count – rows with at least one match
            run_count     – rows where the matcher was actually run (non-None)
    """
    run_count = 0
    matched_count = 0
 
    for v in data_dict.values():
        if v is None:
            # No term annotation → excluded from rate calculation
            continue
        run_count += 1
        if isinstance(v, list) and v:
            matched_count += 1
 
    percentage = (matched_count / run_count * 100) if run_count > 0 else 0.0
    return percentage, matched_count, run_count
 
 
def print_success_rate(term_results, category_name, output_dir="./data/results_analysis",
                       filename="term_accuracy_rates.txt", clear_file=False):
    """
    Calculate and save success rates per model, writing both a human-readable
    .txt file and a TSV file where models are rows and the rate is a column.
 
    Only rows where the matcher was actually run (non-None result) count toward
    the denominator, so missing-term rows do not dilute the reported rate.
 
    Args:
        term_results (dict): {model_name: {sentence_id: list_of_matches | None}}
        category_name (str): Label for this evaluation category
        output_dir (str):    Directory to save output files
        filename (str):      Base name for the .txt file (TSV gets the same stem)
        clear_file (bool):   If True overwrite files; if False append
 
    Returns:
        dict: {model_name: percentage} for each model
    """
 
    # ── Collect per-model stats ──────────────────────────────────────────────
    stats = {}  # model → (percentage, matched, run)
 
    for model_name, result_dict in term_results.items():
        percentage, matched, run = calculate_success_rate(result_dict)
        total = len(result_dict)          # includes skipped (None) rows
        skipped = total - run
        stats[model_name] = {
            "percentage": round(percentage, 2),
            "matched":    matched,
            "run":        run,
            "skipped":    skipped,
            "total":      total,
        }
 
    os.makedirs(output_dir, exist_ok=True)
    txt_mode = "w" if clear_file else "a"
 
    # ── .txt report (human-readable) ────────────────────────────────────────
    txt_path = os.path.join(output_dir, filename)
    with open(txt_path, txt_mode, encoding="utf-8") as f:
        f.write(f"\n{category_name}:\n")
        for model, s in stats.items():
            f.write(
                f"  {model}: {s['percentage']:.2f}%"
                f"  ({s['matched']}/{s['run']} run"
                f", {s['skipped']} skipped / {s['total']} total)\n"
            )
 
    # ── TSV report (models as rows, one column per metric) ──────────────────
    # Use the same stem as the .txt file, with a _rates.tsv suffix so the two
    # files stay together and are easy to load into pandas / Excel.
    tsv_stem = os.path.splitext(filename)[0]
    tsv_path = os.path.join(output_dir, f"{tsv_stem}.tsv")
 
    tsv_header = "category\tmodel\tsuccess_rate_%\tmatched\trun\tskipped\ttotal\n"
    tsv_mode = "w" if clear_file else "a"
 
    with open(tsv_path, tsv_mode, encoding="utf-8") as f:
        # Write header only when starting fresh
        if clear_file:
            f.write(tsv_header)
        for model, s in stats.items():
            f.write(
                f"{category_name}\t{model}\t{s['percentage']:.2f}"
                f"\t{s['matched']}\t{s['run']}\t{s['skipped']}\t{s['total']}\n"
            )
 
    print(f"Success rates for '{category_name}' written to:")
    print(f"  txt → {txt_path}")
    print(f"  tsv → {tsv_path}")
 
    # Return simple {model: percentage} dict to keep the public API stable
    return {model: s["percentage"] for model, s in stats.items()}
 