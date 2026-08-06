import pandas as pd
import spacy
from tqdm import tqdm

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

#makes sure input terms are in the right format or NaN
def is_valid_term_type(val):
    if pd.isna(val):
        return False
    stripped = str(val).strip()
    return stripped not in ('', '[]', '[ ]')
# Function to lemmatize sentences


def lemmatize_sentence(sentence, model):

    nlp = model  # Ensure you have the model downloaded: python -m spacy download en_core_web_sm
    #print(model.pipe_names)
    if pd.isna(sentence):
        return sentence
    else:
        doc = nlp(sentence)
        return ' '.join([token.lemma_ for token in doc])
    
def batch_lemmatize_sentences(sentences, model, batch_size):
    """
    Docstring for batch_lemmatize_sentences
    
    :param sentences: takes list of sentences to lemmatize
    :param model: spacy model init

    :return: list of lemmatized sentences
    """

    nlp = model
    disable_pipes = [
        pipe for pipe in nlp.pipe_names
        if pipe not in {"lemmatizer", "morphologizer", "tagger", "transformer"}
    ]

    # Avoid passing none to the matcher when there is no term to match
    # Replace None with "" as a placeholder, track which indices to restore as None
    none_indices = {i for i, s in enumerate(sentences) if s is None}
    safe_sentences = [s if s is not None else "" for s in sentences]

    output = []
    for i, doc in enumerate(tqdm(
        nlp.pipe(safe_sentences, batch_size=batch_size, disable=disable_pipes, n_process=1),
        total=len(safe_sentences),
    )):
        if i in none_indices:
            output.append(None)   # preserve None → check_type → get_terms_list → signals no term to match
        else:
            output.append(" ".join(token.lemma_ for token in doc))

    return output


def batch_lemmatize_terms(term_cells, model, batch_size):
    """
    Lemmatize comma-separated multi-term cells (e.g. "Internet-Suchmaschine,Online-Suchmaschine").

    Each term is lemmatized on its own, never the whole comma-joined cell,
    so the comma is never fed into the lemmatizer -- avoiding the lemmatizer
    turning "," into a stray artifact (e.g. "--") that swallows the delimiter.

    :param term_cells: list of cell values (str or None), each optionally
                        containing multiple terms separated by ','
    :param model: spacy model init
    :param batch_size: batch size for nlp.pipe

    :return: list of lemmatized cells (str or None), same order/length as term_cells
    """
    nlp = model
    disable_pipes = [
        pipe for pipe in nlp.pipe_names
        if pipe not in {"lemmatizer", "morphologizer", "tagger", "transformer"}
    ]

    split_terms = [
        [t.strip() for t in cell.split(",") if t.strip()] if cell is not None else []
        for cell in term_cells
    ]
    flat_terms = [term for terms in split_terms for term in terms]

    lemmatized_flat = [
        " ".join(token.lemma_ for token in doc)
        for doc in tqdm(
            nlp.pipe(flat_terms, batch_size=batch_size, disable=disable_pipes, n_process=1),
            total=len(flat_terms),
        )
    ]

    output = []
    i = 0
    for cell, terms in zip(term_cells, split_terms):
        if cell is None:
            output.append(None)
            continue
        lemmatized_terms = lemmatized_flat[i:i + len(terms)]
        i += len(terms)
        output.append(",".join(lemmatized_terms))

    return output