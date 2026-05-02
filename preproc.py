# How to call me: 
# python ./preproc.py --testset legistyr --mode hom --lang de
# python ./preproc.py --testset bistro --mode var --lang de

print("I am preprocessing.py")
import os
import glob
from configparser import ConfigParser
import argparse
from tqdm import tqdm
from pathlib import Path

import pandas as pd
import spacy

from utils.preproc_utils import fill_nan_values, conditional_fill_nan_values, lemmatize_sentence, batch_lemmatize_sentences


config = ConfigParser()
config.read('config.ini')

if not config.has_section('main'):
    config.add_section('main')

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Preprocess translation data for different test sets')

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

print('Loading spaCy model...')
if args.lang == 'de':
    model = spacy.load('de_dep_news_trf')
elif args.lang == 'it':   
    model = spacy.load('it_core_news_sm')
else:
    raise ValueError("Unsupported language. Please choose 'de' or 'it'.")

# ============================================================================
# TESTSET-SPECIFIC CONFIGURATION
# ============================================================================

base_dir = Path('./data')

if args.testset == 'legistyr':
    # LegISTyr paths and settings
    
    # Input paths - all modes in legistyr folder
    input_dir = base_dir / 'legistyr' / args.mode
    input_file = f'LegISTyr__{args.mode}.csv'
    delimiter = ';'
    
    # Create directories if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Translation files pattern
    file_pattern = '*.txt'
    
    # Output paths
    output_dir = base_dir / 'preprocessed_texts' / 'legistyr'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessed_file = output_dir / f'preprocessed_texts_{args.mode}.csv'
    raw_file = output_dir / f'raw_texts_{args.mode}.csv'
    
    # Column names (LegISTyr)
    target_hypothesis_col = 'TARGET HYPOTHESIS (DE SOUTH TYROL)'
    other_terms_cols = ['OTHER TERMS SOUTH TYROL (CSV)', 'TERMS FROM OTHER LEGAL SYTEMS (CSV)']
    
    # Output delimiter
    output_delimiter = ';'

elif args.testset == 'bistro':
    # BISTRO paths and settings
    
    # Input paths - all modes in bistro folder
    input_dir = base_dir / 'bistro' / args.mode
    input_file = f'BISTRO__{args.mode}.tsv'
    delimiter = '\t'
    
    # Create directories if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Translation files pattern
    file_pattern = '*.txt'
    
    # Output paths
    output_dir = base_dir / 'preprocessed_texts' / 'bistro'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessed_file = output_dir / f'preprocessed_texts_{args.mode}.tsv'
    raw_file = output_dir / f'raw_texts_{args.mode}.tsv'
    
    # Column names (BISTRO)
    target_hypothesis_col = 'tgt_hypothesis'
    # Will find columns dynamically starting with 'tgt_term_'
    other_terms_cols = None  # Will be populated after loading
    
    # Output delimiter
    output_delimiter = '\t'

# ============================================================================
# LOAD DATA
# ============================================================================

input_path = input_dir / input_file
print(f"Loading data from: {input_path}")

if not input_path.exists():
    raise FileNotFoundError(f"Input file not found: {input_path}")

df = pd.read_csv(input_path, delimiter=delimiter, encoding='utf-8-sig')

# For BISTRO: find columns starting with 'tgt_term_'
if args.testset == 'bistro':
    other_terms_cols = [col for col in df.columns if col.startswith('tgt_term_')]
    print(f"Found {len(other_terms_cols)} target term columns: {other_terms_cols}")

# ============================================================================
# LOAD TRANSLATION FILES
# ============================================================================

print(f"Loading translation files from: {input_dir}")

# Get all matching .txt files
translation_files = glob.glob(str(input_dir / file_pattern))

# Keep track of added columns
new_columns = []

for file_path in translation_files:
    # Get file name without extension to use as column name
    column_name = os.path.splitext(os.path.basename(file_path))[0]

    # Read the file
    with open(file_path, 'r', encoding="utf-8-sig") as f:
        print(f"  Loading: {file_path}")
        translations = f.readlines()

    # Clean newline characters
    translations = [line.strip() for line in translations]

    # Add column
    df[column_name] = translations
    new_columns.append(column_name)

# Replace any leftover newlines
df = df.replace(to_replace=r'\n', value='', regex=True)

# Save model names as environment variable
config.set('main', 'models', ','.join(new_columns))
with open('config.ini', 'w') as configfile:
    config.write(configfile)

# ============================================================================
# PREPROCESSING
# ============================================================================

# Create a copy of the raw data
raw_df = df.copy()

print("Preprocessing target hypothesis...")

# Apply lemmatization to target hypothesis column
df[target_hypothesis_col] = df[target_hypothesis_col].apply(
    lambda sentence: lemmatize_sentence(sentence, model)
)
df[[target_hypothesis_col]] = df[[target_hypothesis_col]].apply(
    lambda col: col.str.replace(r' --', ' ', regex=True)
)

print("Lemmatizing translation columns...")

# Apply lemmatization to new translation columns (batch processing)
for col in tqdm(new_columns, desc="Lemmatizing columns"):
    df[col] = batch_lemmatize_sentences(
        df[col].tolist(),
        model,
        batch_size=100
    )

# Eliminate boilerplate from lemmatization of punctuation
df[new_columns] = df[new_columns].apply(
    lambda col: col.str.replace(r' --', ' ', regex=True)
)

# Clean text in other term columns
if other_terms_cols:
    df[other_terms_cols] = df[other_terms_cols].replace(
        r' -- ', ', ', regex=True
    )

# ============================================================================
# SAVE OUTPUT
# ============================================================================

print(f"Saving preprocessed data to: {preprocessed_file}")
with open(preprocessed_file, 'w', encoding='utf-8-sig') as f:
    df.to_csv(f, index=False, sep=output_delimiter)

print(f"Saving raw data to: {raw_file}")
with open(raw_file, 'w', encoding='utf-8-sig') as f:
    raw_df.to_csv(f, index=False, sep=output_delimiter)

print("✅ Preprocessing complete!")
print(f"   - Preprocessed file: {preprocessed_file}")
print(f"   - Raw file: {raw_file}")
print(f"   - Number of translation columns: {len(new_columns)}")
print(f"   - Column names: {new_columns}")