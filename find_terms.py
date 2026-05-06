
# python ./find_terms.py --testset legistyr --mode hom --lang de
# python ./find_terms.py --testset bistro --mode var --lang de

from configparser import ConfigParser
import argparse
from pathlib import Path

import pandas as pd
import spacy

from spacy.lang.de import German
from spacy.lang.it import Italian
from spacy.matcher import PhraseMatcher

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Find terms in different test sets.')

parser.add_argument('--testset', 
                    type=str,
                    required=True,
                    choices=['legistyr', 'bistro'],
                    help='Test set to use: legistyr or bistro')

parser.add_argument('--mode',
                    type=str,
                    required=True,
                    help='Mode: for legistyr [hom, abbr, simple], for bistro [var, rel, hom]')

parser.add_argument('--lang', 
                    choices=['de', 'it'], 
                    default='de',
                    help='Choose your target language: "de" (Deutsch) or "it" (Italian). Default is "de".')

args = parser.parse_args()

print(f"Testset: {args.testset}")
print(f"Mode: {args.mode}")
print(f"Language: {args.lang}")

# ============================================================================
# VALIDATE MODE FOR TESTSET
# ============================================================================

if args.testset == 'legistyr':
    valid_modes = ['hom', 'abbr', 'simple_terms']
    if args.mode not in valid_modes:
        raise ValueError(f"Invalid mode '{args.mode}' for testset 'legistyr'. Choose from: {valid_modes}")
elif args.testset == 'bistro':
    valid_modes = ['var', 'rel', 'hom']
    if args.mode not in valid_modes:
        raise ValueError(f"Invalid mode '{args.mode}' for testset 'bistro'. Choose from: {valid_modes}")

# ============================================================================
# LOAD SPACY MODEL
# ============================================================================

if args.lang == 'de':
    nlp_lang = German()
    print("German model and matcher loaded.")
elif args.lang == 'it':   
    nlp_lang = Italian()
    print("Italian model and matcher loaded.")
else:
    raise ValueError("Unsupported language. Please choose 'de' or 'it'.")


# Finally importing utils that depend on the language setting
from utils.term_finder_utils import create_entries
from utils.results_utils import save_term_results, find_terms_over_models, print_success_rate

# ============================================================================
# READING CONFIG FOR MODEL NAMES
# ============================================================================

config = ConfigParser()
config.read('config.ini')

models_str = config.get('main', 'models')
models_list = [item.strip() for item in models_str.split(',')]

print(f"Translation models: {models_list}")

# ============================================================================
# TESTSET-SPECIFIC CONFIGURATION
# ============================================================================

base_dir = Path('./data')

if args.testset == 'legistyr':
    # LegISTyr configuration
    input_dir = base_dir / 'preprocessed_texts' / 'legistyr'
    
    preprocessed_file = input_dir / f'preprocessed_texts_{args.mode}.csv'
    raw_file = input_dir / f'raw_texts_{args.mode}.csv'
    delimiter = ';'
    
    # Domains to search (based on mode)
    if args.mode == 'hom':
        domains = ["South-Tyrol", "other_tyrol", "other_systems", "homonym"]
        domain_display_names = {
            "South-Tyrol": "South_Tyrol_terms",
            "other_tyrol": "other_south_tyrol_terms",
            "other_systems": "other_legal_systems_terms",
            "homonym": "wrong_homonyms"
        }
        success_rate_categories = {
            "South-Tyrol": "Success rate of target South-Tyrolean terms",
            "other_tyrol": "Success rate of alternative South-Tyrolean terms",
            "other_systems": "Success rate of terms from extraneous legal systems",
            "homonym": "Percentage of incorrect homonym insertion"
        }
    else:  # simple or abbr
        domains = ["South-Tyrol", "other_tyrol", "other_systems"]
        domain_display_names = {
            "South-Tyrol": "South_Tyrol_terms",
            "other_tyrol": "other_south_tyrol_terms",
            "other_systems": "other_legal_systems_terms"
        }
        success_rate_categories = {
            "South-Tyrol": "Success rate of target South-Tyrolean terms",
            "other_tyrol": "Success rate of alternative South-Tyrolean terms",
            "other_systems": "Success rate of terms from extraneous legal systems"
        }

elif args.testset == 'bistro':
    # BISTRO configuration
    input_dir = base_dir / 'preprocessed_texts' / 'bistro'
    
    preprocessed_file = input_dir / f'preprocessed_texts_{args.mode}.tsv'
    raw_file = input_dir / f'raw_texts_{args.mode}.tsv'
    delimiter = '\t'

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"Loading preprocessed data from: {preprocessed_file}")
print(f"Loading raw data from: {raw_file}")

if not preprocessed_file.exists():
    raise FileNotFoundError(f"Preprocessed file not found: {preprocessed_file}")
if not raw_file.exists():
    raise FileNotFoundError(f"Raw file not found: {raw_file}")

df = pd.read_csv(preprocessed_file, delimiter=delimiter, encoding='utf-8-sig')
raw_df = pd.read_csv(raw_file, delimiter=delimiter, encoding='utf-8-sig')

# For BISTRO: detect columns starting with 'tgt_term_' - these become our domains
if args.testset == 'bistro':
    tgt_term_cols = [col for col in df.columns if col.startswith('tgt_term_')]
    print(f"Detected term columns for BISTRO: {tgt_term_cols}")
    
    # Each tgt_term_* column is a separate domain
    domains = tgt_term_cols
    domain_display_names = {col: col for col in tgt_term_cols}
    success_rate_categories = {col: f"Success rate of {col}" for col in tgt_term_cols}
else:
    tgt_term_cols = None

# ============================================================================
# CREATE ENTRIES
# ============================================================================

print(f"Creating entries for {args.testset} - {args.mode}...")

# For LegISTyr homonyms, pass homonym=True
if args.testset == 'legistyr' and args.mode == 'hom':
    entries_dict = create_entries(df, models_list, homonym=True, testset='legistyr')  # ✅ Added homonym=True
    raw_entries_dict = create_entries(raw_df, models_list, homonym=True, testset='legistyr')
elif args.testset == 'legistyr':
    entries_dict = create_entries(df, models_list, homonym=False, testset='legistyr')  # ✅ Added homonym=False
    raw_entries_dict = create_entries(raw_df, models_list, homonym=False, testset='legistyr')
else:  # bistro
    entries_dict = create_entries(df, models_list, homonym=False, testset='bistro')  # ✅ Added homonym=False
    raw_entries_dict = create_entries(raw_df, models_list, homonym=False, testset='bistro')

# ============================================================================
# FIND TERMS ACROSS ALL DOMAINS
# ============================================================================

all_results = {}

for domain in domains:
    print(f"\n{'='*60}")
    print(f"Finding terms for domain: {domain}")
    print(f"{'='*60}")
    
    term_results = find_terms_over_models(
        nlp_lang, 
        entries_dict, 
        raw_entries_dict, 
        models_list, 
        domain,
        lang=args.lang,
        tgt_term_columns=tgt_term_cols  # Pass BISTRO columns (None for LegISTyr)
    )
    
    all_results[domain] = term_results
    
    for col_name, results_dict in term_results.items():
        print(f"  Model {col_name}: {len(results_dict)} results")

# ============================================================================
# SAVE RESULTS TO CSV
# ============================================================================

print(f"\n{'='*60}")
print("Saving results to CSV...")
print(f"{'='*60}")

output_dir = base_dir / 'results' / args.testset / args.mode

for domain in domains:
    filename = domain_display_names[domain]
    save_term_results(
        all_results[domain], 
        filename=filename,
        output_dir=str(output_dir)
    )

# ============================================================================
# CALCULATE AND SAVE SUCCESS RATES
# ============================================================================

print(f"\n{'='*60}")
print("Calculating success rates...")
print(f"{'='*60}")

analysis_dir = base_dir / 'results_analysis' / args.testset / args.mode
analysis_dir.mkdir(parents=True, exist_ok=True)
analysis_file = analysis_dir / "term_accuracy_rates.txt"

# Clear file for first domain, then append for others
for idx, domain in enumerate(domains):
    category_name = success_rate_categories[domain]
    clear_file = (idx == 0)  # Clear only on first iteration
    
    print_success_rate(
        all_results[domain],
        category_name=category_name,
        output_dir=str(analysis_dir),
        filename="term_accuracy_rates.txt",
        clear_file=clear_file
    )

print(f"\n✅ Term finding complete!")
print(f"   - Results saved to: {output_dir}")
print(f"   - Analysis saved to: {analysis_file}")