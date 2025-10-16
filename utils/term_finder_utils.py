import pandas
import spacy
import re
from pathlib import Path
import json
from typing import List, Tuple


# Load the spacy model

from spacy.matcher import PhraseMatcher
from spacy.lang.de import German

nlp_de = German()
matcher_de = PhraseMatcher(nlp_de.vocab, attr="LOWER")

#import external file with probabilities
# Handling imports and environment variable for the upload of the external file with compound splitter probabilities

NGRAM_PATH = Path.cwd() / "ngram_probs.json"  # Use current working directory

if not NGRAM_PATH.exists():
    raise FileNotFoundError(f"File not found: {NGRAM_PATH}")

with open(NGRAM_PATH) as f:
    ngram_probs = json.load(f)

print("Loaded ngram_probs successfully!")



# This function restructures the data as a tuple with four elements: string, string, list or nan, list or nan
def create_entries(table, translation_columns, homonym = False):
    """
    Creates a dictionary of entries for each translation column.

    Parameters:
    - table (pd.DataFrame): DataFrame with translation and term-related columns.
    - translation_columns (list): List of column names that contain the translations (e.g., new_columns).

    Returns:
    dict: {column_name: list of tuples}, where each tuple contains:
        (translation, target hypothesis [list], alternative options [list or NaN], other term options [list or NaN])
    """
    results = {}

    for col in translation_columns:
        entries = [
            table[col],  # This is the machine-translated sentence
            table['TARGET HYPOTHESIS '].apply(lambda x: [x] if isinstance(x, str) else x),
            table['ALTRE OPZIONI STAA (CSV)'].apply(lambda x: x.split(", ") if isinstance(x, str) else x),
            table['TERMINI ALTRI ORDINAMENTI (CSV)'].apply(lambda x: x.split(", ") if isinstance(x, str) else x)
        ]

        # Add 'options' column if include_options is True
        if homonym:
            entries.append(table['OPTIONS'].apply(lambda x: x.split(", ") if isinstance(x, str) else x))

        entries = list(zip(*entries))
        results[col] = entries

    return results
    
    
# This class finds terms in a sentence

# Constructing the class

class TermFinder:

    def __init__(self, nlp_model, entry_list):
        """
        Initialize the TermMatcher class.

        Args:
            nlp_model: A SpaCy language model instance.
        """
        self.nlp = nlp_model
        self.entry_list = entry_list


    def check_type(self, terms_list):
        """Check data type and ensure it is List"""
        return isinstance(terms_list, list)


    def split_compound(self, word: str) -> List[Tuple[float, str, str]]:
        """Return list of possible splits, best first.
        :param word: Word to be split
        :return: List of all splits
        """
        word = word.lower()

        # If there is a hyphen in the word, return part of the word behind the last hyphen
        if '-' in word:
            return [(1., re.search('(.*)-', word.title()).group(1), re.sub('.*-', '', word.title()))]

        scores = list() # Score for each possible split position

        # Iterate through characters, start at forth character, go to 3rd last
        for n in range(3, len(word)-2):
            pre_slice = word[:n]

            # Cut of Fugen-S
            if pre_slice.endswith('ts') or pre_slice.endswith('gs') or pre_slice.endswith('ks') \
                    or pre_slice.endswith('hls') or pre_slice.endswith('ns'):
                if len(word[:n-1]) > 2: pre_slice = word[:n-1]

            # Start, in, and end probabilities
            pre_slice_prob = list()
            in_slice_prob = list()
            start_slice_prob = list()

            # Extract all ngrams
            for k in range(len(word)+1, 2, -1):

                # Probability of first compound, given by its ending prob
                if not pre_slice_prob and k <= len(pre_slice):
                    # The line above deviates from the description in the thesis;
                    # it only considers word[:n] as the pre_slice.
                    # This improves accuracy on GermEval and increases speed.
                    # Use the line below to replicate the original implementation:
                    # if k <= len(pre_slice):
                    end_ngram = pre_slice[-k:]  # Look backwards
                    pre_slice_prob.append(ngram_probs["suffix"].get(end_ngram, -1))   # Punish unlikely pre_slice end_ngram

                # Probability of ngram in word, if high, split unlikely
                in_ngram = word[n:n+k]
                in_slice_prob.append(ngram_probs["infix"].get(in_ngram, 1)) # Favor ngrams not occurring within words

                # Probability of word starting
                # The condition below deviates from the description in the thesis (see above comments);
                # Remove the condition to restore the original implementation.
                if not start_slice_prob:
                    ngram = word[n:n+k]
                    # Cut Fugen-S
                    if ngram.endswith('ts') or ngram.endswith('gs') or ngram.endswith('ks') \
                            or ngram.endswith('hls') or ngram.endswith('ns'):
                        if len(ngram[:-1]) > 2:
                            ngram = ngram[:-1]

                    start_slice_prob.append(ngram_probs["prefix"].get(ngram, -1))

            if not pre_slice_prob or not start_slice_prob:
                continue

            start_slice_prob = max(start_slice_prob)
            pre_slice_prob = max(pre_slice_prob)  # Highest, best pre_slice
            in_slice_prob = min(in_slice_prob)  # Lowest, punish splitting of good in_grams
            score = start_slice_prob - in_slice_prob + pre_slice_prob
            scores.append((score, word[:n].title(), word[n:].title()))

        scores.sort(reverse=True)

        if not scores:
            scores = [[0, word.title(), word.title()]]

        return sorted(scores, reverse = True)


    def phrase_matcher(self, sentence, list_of_terms):
        """
        Match terms in a sentence using PhraseMatcher.
        
        Args:
            sentence: The sentence to search in
            list_of_terms: List of terms to search for
            
        Returns:
            List of matched spans
        """
        # Create a fresh matcher with LOWER attribute for case-insensitive matching
        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        # Filter out empty or invalid terms
        valid_terms = [term for term in list_of_terms if term and isinstance(term, str) and term.strip()]
        
        if not valid_terms:
            return []

        pattern_de = [self.nlp.make_doc(term) for term in valid_terms]
        match_id_str = "TERM_MATCH"

        matcher.add(match_id_str, pattern_de)

        doc = self.nlp(sentence)
        matches = matcher(doc, as_spans=True)

        return matches



    def find_terms(self, domain, homonym=False):
        """
        Find terms in sentences based on the specified domain.
        
        Args:
            domain: One of "South-Tyrol", "other_tyrol", "other_systems" or "homonym"
            
        Returns:
            Dictionary mapping sentences to their matched terms
        """
        results = {}
        for sent, term, other_term_list, other_system_list, *homonym_list in self.entry_list:      

                # Skip if sentence is None or empty
            if not sent or not isinstance(sent, str):
                    results[sent] = []
                    continue

            # Determine which terms list to use based on domain
            if domain == "South-Tyrol":
                if self.check_type(term):
                    terms_list = list(term)
                else:
                    terms_list = []

            elif domain == "other_tyrol":
                if self.check_type(other_term_list):
                    terms_list = other_term_list
                else:
                    terms_list = []
                    
            elif domain == "other_systems":
                if self.check_type(other_system_list):
                    terms_list = other_system_list
                else:
                    terms_list = []

            #Further check to keep only the wrong homonym among the term options
            elif domain == "homonym":
                if self.check_type(homonym_list):
                    raw_terms_list = homonym_list[0]

                    term_str = term if isinstance(term, str) else str(term)
                    terms_list = [h for h in raw_terms_list if h not in term_str]

                else:
                    terms_list = []
                    
            else:
                raise Exception("Invalid argument. You must choose a domain among 'South-Tyrol', 'other_tyrol', 'other_systems' or 'homonym'")
    
            # If terms_list is empty or contains only invalid values, skip
            if not terms_list:
                results[sent] = []
                continue
            
            # Try to find terms with spacy
            pattern_match = self.phrase_matcher(sent, terms_list)
            #print(f"Sentence: {sent[:50]}... | Terms: {terms_list} | Matches: {[m.text for m in pattern_match]}")
            
            if len(pattern_match) == 0:  # if no match found, try again with compound split
                #print(f"  -> No direct match, trying compound splitter...")

                # Split compounds
                split_sent = " ".join(" ".join(self.split_compound(word)[0][1:]) for word in sent.split())
                split_terms = [" ".join(" ".join(self.split_compound(word)[0][1:]) for word in t.split()) for t in terms_list]

                # Lemmatize split sentence and terms
                lemmatized_sent = ' '.join([token.lemma_ for token in self.nlp(split_sent)])
                lemmatized_terms = [' '.join([token.lemma_ for token in self.nlp(split_term.strip())]) for split_term in split_terms]

                split_pattern_match = self.phrase_matcher(lemmatized_sent, lemmatized_terms)
                
                #print(f"  -> After splitting: {[m.text for m in split_pattern_match]}")

                results[sent] = split_pattern_match  # term found after compound splitting

            else:
                results[sent] = pattern_match  # term found after normal spacy matching

        return results
    

