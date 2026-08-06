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
import time

import pandas as pd
import spacy

from utils.preproc_utils import batch_lemmatize_sentences, batch_lemmatize_terms, is_valid_term_type


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
                    help='Mode: for legistyr [hom, abbr, simple_terms], for bistro [var, rel, hom]')

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

start = time.time()
print("Starting preprocessing...")

base_dir = Path('./data')

if args.testset == 'legistyr':
    # LegISTyr paths and settings
    input_dir = base_dir / 'legistyr' / args.mode
    input_file = f'LegISTyr__{args.mode}.csv'
    delimiter = ';'
    output_delimiter = ';'
    
    # All columns to lemmatize
    term_cols = [
        'TARGET HYPOTHESIS (DE SOUTH TYROL)',
        'OTHER TERMS SOUTH TYROL (CSV)',
        'TERMS FROM OTHER LEGAL SYTEMS (CSV)'
    ]
        # Homonym mode: add OPTIONS column for lemmatization
    if args.mode == 'hom':
        term_cols.append('OPTIONS')

elif args.testset == 'bistro':
    # BISTRO paths and settings
    input_dir = base_dir / 'bistro' / args.mode
    input_file = f'BISTRO__{args.mode}.tsv'
    delimiter = '\t'
    output_delimiter = '\t'
    
    # Will be populated after loading
    term_cols = []

# Common settings
input_dir.mkdir(parents=True, exist_ok=True)
file_pattern = '*.txt'

output_dir = base_dir / 'preprocessed_texts' / args.testset
output_dir.mkdir(parents=True, exist_ok=True)

file_ext = 'csv' if args.testset == 'legistyr' else 'tsv'
preprocessed_file = output_dir / f'preprocessed_texts_{args.mode}.{file_ext}'
raw_file = output_dir / f'raw_texts_{args.mode}.{file_ext}'

# ============================================================================
# LOAD DATA
# ============================================================================

input_path = input_dir / input_file
print(f"Loading data from: {input_path}")

if not input_path.exists():
    raise FileNotFoundError(f"Input file not found: {input_path}")

df = pd.read_csv(input_path, delimiter=delimiter, encoding='utf-8-sig')

# For BISTRO: dynamically find term columns
if args.testset == 'bistro':
    term_cols = [col for col in df.columns if col.startswith('tgt_term_')]
    print(f"Found {len(term_cols)} target term columns: {term_cols}")

# ============================================================================
# LOAD TRANSLATION FILES
# ============================================================================

print(f"Loading translation files from: {input_dir}")

translation_files = glob.glob(str(input_dir / file_pattern))
new_columns = []

for file_path in translation_files:
    column_name = os.path.splitext(os.path.basename(file_path))[0]
    
    with open(file_path, 'r', encoding="utf-8-sig") as f:
        print(f"  Loading: {file_path}")
        translations = [line.strip() for line in f.readlines()]
    
    df[column_name] = translations
    new_columns.append(column_name)

# Clean any leftover newlines
df = df.replace(to_replace=r'\n', value='', regex=True)

# Save model names to config
config.set('main', 'models', ','.join(new_columns))
with open('config.ini', 'w') as configfile:
    config.write(configfile)

# ============================================================================
# PREPROCESSING
# ============================================================================

# Create a copy of the raw data
raw_df = df.copy()

# Lemmatize translation columns from .txt files
print("Lemmatizing translation columns...")
for col in tqdm(new_columns, desc="Lemmatizing translations"):
    df[col] = batch_lemmatize_sentences(
        df[col].tolist(),
        model,
        batch_size=1000
    )

# Lemmatize term columns (hypothesis + terms for LegISTyr, tgt_term_* for BISTRO)
if term_cols:
    print(f"Lemmatizing term columns: {term_cols}")
    for col in tqdm(term_cols, desc="Lemmatizing terms"):
        df[col] = batch_lemmatize_terms(
            [val if is_valid_term_type(val) else None for val in df[col].tolist()],
            model,
            batch_size=1000
        )

# Clean up ' --' from lemmatization artifacts in all lemmatized columns
all_lemmatized_cols = new_columns + term_cols
df[all_lemmatized_cols] = df[all_lemmatized_cols].apply(
    lambda col: col.str.replace(r' --', ' ', regex=True)
)

end = time.time()
print(f"Elapsed: {end - start:.2f} seconds")

# ============================================================================
# SAVE OUTPUT
# ============================================================================

print(f"Saving preprocessed data to: {preprocessed_file}")
df.to_csv(preprocessed_file, index=False, sep=output_delimiter, encoding='utf-8-sig')

print(f"Saving raw data to: {raw_file}")
raw_df.to_csv(raw_file, index=False, sep=output_delimiter, encoding='utf-8-sig')

print("✅ Preprocessing complete!")
print(f"   - Preprocessed file: {preprocessed_file}")
print(f"   - Raw file: {raw_file}")
print(f"   - Number of translation columns: {len(new_columns)}")
print(f"   - Translation columns: {new_columns}")
if term_cols:
    print(f"   - Lemmatized term columns: {term_cols}")