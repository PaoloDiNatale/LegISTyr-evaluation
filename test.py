print("okay")
print("This is test.py")
import pandas as pd
import spacy

from utils.preproc_utils import fill_nan_values, conditional_fill_nan_values, lemmatize_sentence


import faulthandler
faulthandler.enable()

print('I am alive!')

# Load the data
df = pd.read_csv('data/Test_set_MT_terminology.csv', delimiter=';', encoding='utf-8-sig')

# Drag on term id and target term fields to fill in empty values
dragged_columns = ['TARGET HYPOTHESIS ', 'N. TERMINE'] # Term id and target term fields

for column in dragged_columns:
    df = fill_nan_values(df, column) # Fill NaN values in the specified column

# Drag on alternative terms fields conditioned on the term id
conditional_dragged_columns = ['ALTRE OPZIONI STAA (CSV)', 'TERMINI ALTRI ORDINAMENTI (CSV)'] # Alternative terms fields
ref_column = 'N. TERMINE'
for conditional_dragged_column in conditional_dragged_columns:
    df = conditional_fill_nan_values(df, conditional_dragged_column, ref_column) # Fill NaN values in the specified column conditionally

# Select only relevant columns
df = df[['N. TERMINE', 'ESEMPIO IT', 'TARGET HYPOTHESIS ', 'ALTRE OPZIONI STAA (CSV)', 'TERMINI ALTRI ORDINAMENTI (CSV)']]

print('Loaded data and filled in missing values. Now starts to get translations') 

# Load translations from txt file
with open('data/translations.txt', 'r', encoding='utf-8-sig') as f:
    translations = f.readlines()

# Add translations as column in df
df.insert(2, 'Machine translation', translations) # Insert translations as the third column
df = df.replace(to_replace=r'\n', value='', regex=True) # Remove newline characters

print('starts to lemmatize now...')

# lemmatize all sentences and words in the DataFrame
df.iloc[:, 2:] = df.iloc[:, 2:].map(lemmatize_sentence)

print('lemmatization completed')