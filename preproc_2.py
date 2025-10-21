print("I am preprocessing.py")
import os
import glob
from configparser import ConfigParser
import argparse

import pandas as pd
import spacy
#import spacy_transformers

from utils.preproc_utils import fill_nan_values, conditional_fill_nan_values, lemmatize_sentence


config = ConfigParser()
config.read('config.ini')

if not config.has_section('main'):
    config.add_section('main')
#import faulthandler
#faulthandler.enable()

parser = argparse.ArgumentParser(description='Create entries from translation table')

parser.add_argument('--hom', action="store_true",
                       help='Include if you are testing on the honomym subset')
    
args = parser.parse_args()

print('I am alive!')

#import testset
if args.hom:
    df = pd.read_csv('./data/homonyms/1_testset_omonimi.csv', delimiter=';', encoding='utf-8-sig')
else:
    df = pd.read_csv('./data/simple_terms/testset_simple_terms.csv', delimiter=';', encoding='utf-8-sig')


# Folder containing the translation files
if args.hom:
    folder_path = './data/homonyms'
    file_pattern = '*.txt'
else: 
    folder_path = './data/simple_terms'
    file_pattern = '*.txt'

print("using source folder " + folder_path)

# Get all matching .txt files
translation_files = glob.glob(os.path.join(folder_path, file_pattern))

# Keep track of added columns
new_columns = []

for file_path in translation_files:
    # Get file name without extension to use as column name
    column_name = os.path.splitext(os.path.basename(file_path))[0]

    # Read the file
    with open(file_path, 'r', encoding="ISO-8859-1") as f:
        translations = f.readlines()

    # Clean newline characters
    translations = [line.strip() for line in translations]

    # Add column
    df[column_name] = translations
    new_columns.append(column_name)

# Replace any leftover newlines (in case)
df = df.replace(to_replace=r'\n', value='', regex=True)

#save model names as env variable
config.set('main', 'models', ','.join(new_columns))
with open('config.ini', 'w') as configfile:
    config.write(configfile)


    # Apply lemmatization to new translation columns
for col in new_columns:
    df[col] = df[col].apply(lemmatize_sentence)

# Eliminate boilerplate from lemmatization of punctuation from translations
df[new_columns] = df[new_columns].apply(lambda col: col.str.replace(r' --', ' ', regex=True))

#clean text in other columns
df[['ALTRE OPZIONI STAA (CSV)', 'TERMINI ALTRI ORDINAMENTI (CSV)']] = df[['ALTRE OPZIONI STAA (CSV)', 'TERMINI ALTRI ORDINAMENTI (CSV)']].replace(r' -- ', ', ', regex=True)


if args.hom:
    preprocessed_file = 'data/preprocessed_data_homs.csv'
else:
    preprocessed_file = 'data/preprocessed_data_simple_terms.csv'

# Save the preprocessed DataFrame to a new CSV file
with open(preprocessed_file, 'w', encoding='utf-8-sig') as f:
    df.to_csv(f, index=False, sep=";")

print("I'm done")