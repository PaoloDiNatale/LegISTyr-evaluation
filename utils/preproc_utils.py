import pandas as pd
import spacy

# drag term id and target term fields
def fill_nan_values(df, column_name):
    """
    Fills NaN values in a specified column of the DataFrame by dragging down the last valid value.
    
    Parameters:
    - df: pandas DataFrame
    - column_name: string, the column to perform the operation on
    
    Returns:
    - Modified DataFrame with NaN values filled
    """
    last_valid_value = None
    
    for idx in range(len(df)):
        if pd.notna(df.loc[idx, column_name]):
            last_valid_value = df.loc[idx, column_name]
        elif last_valid_value is not None:
            df.loc[idx, column_name] = last_valid_value
            
    return df


# drag alternative terms fields conditioned on the term id
def conditional_fill_nan_values(df, target_column, reference_column):
    """
    Fills NaN values in a specified target column of the DataFrame by dragging down the last valid value,
    only if the value of the reference column remains the same. If the reference column value changes,
    dragging stops and resumes with a new valid value.
    
    Parameters:
    - df: pandas DataFrame
    - target_column: string, the column to fill NaN values in
    - reference_column: string, the column to check for value changes
    
    Returns:
    - Modified DataFrame with NaN values in the target column filled conditionally
    """
    last_valid_value = None
    last_reference_value = None

    for idx in range(len(df)):
        current_reference_value = df.loc[idx, reference_column]
        
        if pd.notna(df.loc[idx, target_column]):
            # Update last valid value and reference value when target is not NaN
            last_valid_value = df.loc[idx, target_column]
            last_reference_value = current_reference_value
        elif last_reference_value == current_reference_value:
            # Drag down the last valid value if the reference column hasn't changed
            if last_valid_value is not None:
                df.loc[idx, target_column] = last_valid_value
        else:
            # Stop dragging if the reference column value changes
            last_valid_value = None
            last_reference_value = current_reference_value
    
    return df


# Function to lemmatize sentences
def lemmatize_sentence(sentence, model):

    nlp = model  # Ensure you have the model downloaded: python -m spacy download en_core_web_sm

    if pd.isna(sentence):
        return sentence
    else:
        doc = nlp(sentence)
        return ' '.join([token.lemma_ for token in doc])