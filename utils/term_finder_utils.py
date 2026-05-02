import pandas
import spacy
import re
from pathlib import Path
import json
import re
from typing import List, Tuple
from functools import wraps
from utils.config.config import get_lang


# Load the spacy model

from spacy.matcher import PhraseMatcher
from spacy.lang.de import German

nlp_de = German()
matcher_de = PhraseMatcher(nlp_de.vocab, attr="LOWER")

def language_check(lang_code):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if get_lang() != lang_code:
                return []  # skip and return empty list
            return func(*args, **kwargs)
        return wrapper
    return decorator


#import external file with probabilities
# Handling imports and environment variable for the upload of the external file with compound splitter probabilities
lang = get_lang()
print(f"Current language in term_finder_utils: {lang}")
if lang == "de":
    NGRAM_PATH = Path.cwd() / "ngram_probs.json"  # Use current working directory

    if not NGRAM_PATH.exists():
        raise FileNotFoundError(f"File not found: {NGRAM_PATH}")

    with open(NGRAM_PATH) as f:
        ngram_probs = json.load(f)

    print("Loaded ngram_probs successfully!")

else:
    print("No ngram probabilities needed for Italian.")



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
            table['TARGET HYPOTHESIS (DE SOUTH TYROL)'].apply(lambda x: [x] if isinstance(x, str) else x),
            table['OTHER TERMS SOUTH TYROL (CSV)'].apply(lambda x: x.split(", ") if isinstance(x, str) else x),
            table['TERMS FROM OTHER LEGAL SYTEMS (CSV)'].apply(lambda x: x.split(", ") if isinstance(x, str) else x)
        ]

        # Add 'options' column if homonym is True
        if homonym:
            entries.append(table['OPTIONS'].apply(lambda x: x.split(", ") if isinstance(x, str) else x))

        entries = list(zip(*entries))
        results[col] = entries

    return results
    
    
# Wrapper to simulate spacy text attribute
class _TextAttr:
    def __init__(self, text):
        self.text = text


# This class finds terms in a sentence

class TermFinder:

    def __init__(self, nlp_model, entry_list, raw_entry_list):
        """
        Initialize the TermMatcher class.

        Args:
            nlp_model: A SpaCy language model instance.
        """
        self.nlp = nlp_model
        self.entry_list = entry_list
        self.raw_entry_list = raw_entry_list #Initialize raw_list
        

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


    def return_spans(self, sentence, list_of_terms):
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

        # Extract spans from matches 
        result = []
        for match_id, start, end in matches:
            span = doc[start:end]
            result.append(span)

        return result


    @language_check("de")
    def _compound_split_matcher(self, sent: str, terms_list: list[str]):
        # Split compounds in sentence and terms
        split_sent = " ".join(
            " ".join(self.split_compound(word)[0][1:]) for word in sent.split()
        )
        split_terms = [
            " ".join(" ".join(self.split_compound(word)[0][1:]) for word in t.split())
            for t in terms_list
        ]

        # Lemmatize
        lemmatized_sent = " ".join(token.lemma_ for token in self.nlp(split_sent))
        lemmatized_terms = [
            " ".join(token.lemma_ for token in self.nlp(split_term.strip()))
            for split_term in split_terms
        ]

        # Match again
        split_match = self.phrase_matcher(lemmatized_sent, lemmatized_terms)

        return split_match
    

    # Prepares the term lists for the matcher in find_terms()
    def get_terms_list(self, domain, term, other_term_list, other_system_list, homonym_list):
        if domain == "South-Tyrol":
            return list(term) if self.check_type(term) else []

        elif domain == "other_tyrol":
            return other_term_list if self.check_type(other_term_list) else []

        elif domain == "other_systems":
            return other_system_list if self.check_type(other_system_list) else []

        elif domain == "homonym":
            if self.check_type(homonym_list):
                raw = homonym_list[0]
                term_str = term if isinstance(term, str) else str(term)
                return [h for h in raw if h not in term_str]
            else:
                return []

        else:
            raise Exception(
                "Invalid argument. Choose 'South-Tyrol', 'other_tyrol', 'other_systems', or 'homonym'."
            )
                
    def find_terms(self, domain, homonym=False):
        """
        Find terms in sentences.

        Args:
            domain: domain to search in ("South-Tyrol", "other_tyrol", "other_systems", "homonym")

        Returns:
            Dictionary mapping sentences (idx, processed_sent) to their matched terms
        """
        results = {}

        # Iterate processed and raw entries in parallel
        for idx, (proc_entry, raw_entry) in enumerate(zip(self.entry_list, self.raw_entry_list)):
            # Unpack processed
            sent, term, other_term_list, other_system_list, *homonym_list = proc_entry
            # Unpack raw
            raw_sent, raw_term, raw_other_term_list, raw_other_system_list, *raw_homonym_list = raw_entry

            sent_id = (idx, sent)

            # Ensure strings
            sent_str = sent if isinstance(sent, str) else ""
            raw_sent_str = raw_sent if isinstance(raw_sent, str) else ""

            # Define the lists of terms for homonym and simple term matches
            terms_list = self.get_terms_list(domain, term, other_term_list, other_system_list, homonym_list)
            raw_terms_list = self.get_terms_list(domain, raw_term, raw_other_term_list, raw_other_system_list, raw_homonym_list)


            # Now actually matches terms
            # Try matching; else always assign an empty list
            matches = []

            # Match 1: raw terms and sentence
            if raw_terms_list and raw_sent_str:
                pattern_match = self.phrase_matcher(raw_sent_str, raw_terms_list)

                matches = [
                    m for m in pattern_match
                    if m and str(m).strip() and str(m).lower() != "nan"
                ]

            # match 2: Try matching with lemmatized terms and sentence. Access only if no raw match is found.
            if not matches and terms_list and sent_str:
                pattern_match = self.phrase_matcher(sent_str, terms_list)
                #match 3
                if not pattern_match:
                    pattern_match = self._compound_split_matcher(sent_str, terms_list)

                matches = [
                    m for m in pattern_match
                    if m and str(m).strip() and str(m).lower() != "nan"
                ]



            # Match #4: subsequence-based fuzzy match with 3-letter tolerance for inflection variation ===
                        # Accept a term match if term is a subsequence of the sentence, or if the only
                        # missing characters (up to 3) commonly form inflectional morphemes {e,n,s,r,m,i}.

            infl_chars = set("ensrmi")  # allowed-tolerance characters

            def tokenize_words(s: str):
                return re.findall(r"\w+", s.lower())

            def subseq_match_word_strict(cand_word: str, sent_word: str, infl_chars=infl_chars):
                """
                Strict subsequence match:
                - returns (ok, matched_subseq, missing, extra)
                - missing: chars of cand_word not found in sent_word
                - extra: chars in sent_word that were skipped while matching cand_word (including tail)
                Acceptance rule:
                - total_diff = len(missing) + len(extra) <= 3
                - AND set(missing + extra) is subset of infl_chars (no outside chars allowed)
                """
                j = 0
                matched = []
                missing = []
                extra = []

                for ch in cand_word:
                    found = False
                    while j < len(sent_word):
                        if sent_word[j] == ch:
                            matched.append(ch)
                            j += 1
                            found = True
                            break
                        # any char we skip in sent_word is "extra"
                        extra.append(sent_word[j])
                        j += 1

                    if not found:
                        missing.append(ch)

                # remaining tail in sent_word counts as extra
                if j < len(sent_word):
                    extra.extend(list(sent_word[j:]))

                total_diff = len(missing) + len(extra)

                # strict condition: total <= 3 and every differing char must be in infl_chars
                if total_diff <= 3 and all(c in infl_chars for c in (missing + extra)):
                    return True, "".join(matched), "".join(missing), "".join(extra)
                return False, "".join(matched), "".join(missing), "".join(extra)


            # Example integration into your existing fuzzy loop (word-by-word)
            def fuzzy_match_term_wordwise_strict(cand_term: str, raw_sent_str: str,
                                                min_word_len=10):
                """
                Requires each candidate word to match some sentence word under strict rules.
                Returns (ok, debug_list) where debug_list contains tuples:
                (cand_word, matched_sent_word, matched_subseq, missing, extra)
                """
                sent_words = tokenize_words(raw_sent_str)
                cand_words = tokenize_words(cand_term)

                debug = []

                for cw in cand_words:
                    if len(cw) < min_word_len:
                        return False, debug

                    matched_for_cw = None
                    for sw in sent_words:
                        ok, matched_subseq, missing, extra = subseq_match_word_strict(cw, sw, infl_chars)
                        if ok:
                            matched_for_cw = (cw, sw, matched_subseq, missing, extra)
                            break

                    if not matched_for_cw:
                        return False, debug

                    debug.append(matched_for_cw)

                return True, debug


            # Replace your fuzzy-matching block with something like this:
            if not matches and raw_terms_list and raw_sent_str:
                fuzzy_matches = []
                debug_fuzzy = []

                for cand_term in raw_terms_list:
                    if not isinstance(cand_term, str) or not cand_term:
                        continue

                    ok, dbg = fuzzy_match_term_wordwise_strict(cand_term, raw_sent_str)
                    if ok:
                        fuzzy_matches.append(_TextAttr(cand_term))
                        # dbg: list of (cand_word, sent_word, matched_subseq, missing, extra)
                        debug_fuzzy.append((cand_term, dbg))

                if fuzzy_matches:
                    output_path = "fuzzy_terms.txt"
                    with open(output_path, "a", encoding="utf-8") as f:
                        for term, dbg in debug_fuzzy:
                            parts = []
                            for cw, sw, subseq, missing, extra in dbg:
                                parts.append(f"{cw}->{sw}:{subseq}|missing:{missing}|extra:{extra}")
                            f.write(f"{term}\t" + " || ".join(parts) + "\n")
                    matches = fuzzy_matches

            results[sent_id] = matches

        return results


