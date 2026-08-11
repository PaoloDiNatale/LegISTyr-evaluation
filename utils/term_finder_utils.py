#term_finder_utils_2.py

import pandas
import spacy
import re
from pathlib import Path
import json
import re
from typing import List, Tuple, Optional
from functools import wraps
from functools import lru_cache
from tqdm import tqdm

# Remove the language_check decorator - no longer needed
# from utils.config.config import get_lang

# Load the spacy model
from spacy.matcher import PhraseMatcher
from spacy.lang.de import German

nlp_de = German()
matcher_de = PhraseMatcher(nlp_de.vocab, attr="LOWER")

# DON'T load ngram_probs at module level - do it lazily in TermFinder



# This function restructures the data as a tuple with four elements: string, string, list or nan, list or nan
def create_entries(table, translation_columns, homonym=False, testset='legistyr'):
    """
    Creates a dictionary of entries for each translation column.

    Parameters:
    - table (pd.DataFrame): DataFrame with translation and term-related columns.
    - translation_columns (list): List of column names that contain the translations (e.g., new_columns).
    - homonym (bool): Whether to include homonym options column
    - testset (str): 'legistyr' or 'bistro' - determines column naming convention

    Returns:
    dict: {column_name: list of tuples}, where each tuple contains:
        For legistyr: (translation, target hypothesis [list], alternative options [list or NaN], 
                      other term options [list or NaN], [homonym options if homonym=True])
        For bistro: (translation, tgt_hypothesis [list], tgt_term_1 [list], tgt_term_2 [list], ...)
    """
    results = {}

    for col in translation_columns:
        if testset == 'legistyr':
            # LegISTyr: fixed column names
            entries = [
                table[col],  # Machine-translated sentence
                table['TARGET HYPOTHESIS (DE SOUTH TYROL)'].apply(lambda x: [x] if isinstance(x, str) else x),
                table['OTHER TERMS SOUTH TYROL (CSV)'].apply(lambda x: x.split(", ") if isinstance(x, str) else x),
                table['TERMS FROM OTHER LEGAL SYTEMS (CSV)'].apply(lambda x: x.split(", ") if isinstance(x, str) else x)
            ]

            # Add 'OPTIONS' column if homonym is True
            if homonym:
                entries.append(table['OPTIONS'].apply(lambda x: x.split(", ") if isinstance(x, str) else x))

        elif testset == 'bistro':
            # BISTRO: dynamic columns starting with 'tgt_term_'
            tgt_term_cols = [c for c in table.columns if c.startswith('tgt_term_')]
            
            entries = [
                table[col],  # Machine-translated sentence
                #table['tgt_hypothesis'].apply(lambda x: [x] if isinstance(x, str) else x)
            ]
            
            # Add all tgt_term_* columns; guard against empty lists
            for tgt_col in tgt_term_cols:
                entries.append(
                    table[tgt_col].apply(
                        lambda x: [t.strip() for t in x.split(",") if t.strip() not in ('', '[]', '[ ]')]
                        if isinstance(x, str) and x.strip() not in ('', '[]', '[ ]')
                        else x
                    )
                )
        
        else:
            raise ValueError(f"Unknown testset: {testset}. Choose 'legistyr' or 'bistro'")

        entries = list(zip(*entries))
        results[col] = entries

    return results
    
    
# Wrapper to simulate spacy text attribute
class _TextAttr:
    def __init__(self, text):
        self.text = text


#defines fuzzy term matching

class FuzzyMatcher:
    """
    Fuzzy matching for German inflectional variants using strict subsequence matching.
    Characters must appear in the same order in both words.
    """
    
    # Allowed inflectional characters for German
    INFL_CHARS = frozenset("ensrmi")
    
    # Regex compiled once for performance
    WORD_TOKENIZER = re.compile(r"\w+")
    
    def __init__(self, min_word_len: int = 10, max_diff: int = 3, lang: str = "de"):
        """
        Args:
            min_word_len: Minimum word length to consider for fuzzy matching
            max_diff: Maximum character differences allowed (missing + extra)
            lang: Language code ('de' or 'it')
        """
        self.min_word_len = min_word_len
        self.max_diff = max_diff
        self.lang = lang  # Explicit parameter, not global state
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def tokenize_words(text: str) -> Tuple[str, ...]:
        """Cached tokenization - returns tuple for hashability."""
        return tuple(FuzzyMatcher.WORD_TOKENIZER.findall(text.lower()))
    
    def subseq_match_word(self, cand_word: str, sent_word: str) -> Tuple[bool, str, str, str]:
        """
        Strict subsequence matching - characters must appear IN ORDER.
        
        Args:
            cand_word: The candidate term word
            sent_word: The word from the sentence
            
        Returns:
            (is_match, matched_subseq, missing_chars, extra_chars)
        """
        cand_word = cand_word.lower()
        sent_word = sent_word.lower()
        
        j = 0
        matched = []
        missing = []
        extra = []
        
        # Find subsequence - characters must be in order
        for ch in cand_word:
            found = False
            while j < len(sent_word):
                if sent_word[j] == ch:
                    matched.append(ch)
                    j += 1
                    found = True
                    break
                # Chars we skip in sent_word are "extra"
                extra.append(sent_word[j])
                j += 1
            
            if not found:
                missing.append(ch)
        
        # Remaining tail in sent_word counts as extra
        if j < len(sent_word):
            extra.extend(list(sent_word[j:]))
        
        total_diff = len(missing) + len(extra)
        
        # Condition: total <= max_diff and every differing char must be inflectional
        if total_diff <= self.max_diff and all(c in self.INFL_CHARS for c in (missing + extra)):
            return True, "".join(matched), "".join(missing), "".join(extra)
        
        return False, "".join(matched), "".join(missing), "".join(extra)
    
    def find_fuzzy_matches(self, term: str, sentence: str) -> Optional[Tuple[str, List[dict]]]:
        """
        Find fuzzy matches for a term in a sentence using subsequence matching.
        Each word in the term must match some word in the sentence.
        Only works for German language.
        
        Args:
            term: The term to search for (can be multi-word)
            sentence: The sentence to search in
        
        Returns:
            (matched_term, debug_info) if match found, else None
            
        Examples:
            Term: "öffentliche Verwaltung"
            Sentence: "Die öffentlichen Verwaltungen sind zuständig"
            → MATCH: "öffentliche"→"öffentlichen", "Verwaltung"→"Verwaltungen"
        """
        # Early return for non-German languages
        if self.lang != "de":
            return None
        
        if not term or not isinstance(term, str):
            return None
        
        sent_words = self.tokenize_words(sentence)
        term_words = self.tokenize_words(term)
        
        if not term_words:
            return None
        
        debug_info = []
        
        # Each term word must match SOME sentence word
        for term_word in term_words:
            # Skip short words
            if len(term_word) < self.min_word_len:
                return None
            
            # Try to find a matching sentence word
            found_match = False
            
            for sent_word in sent_words:
                is_match, matched, missing, extra = self.subseq_match_word(term_word, sent_word)
                
                if is_match:
                    debug_info.append({
                        'term_word': term_word,
                        'matched_word': sent_word,
                        'matched_subseq': matched,
                        'missing': missing,
                        'extra': extra,
                        'total_diff': len(missing) + len(extra)
                    })
                    found_match = True
                    break
            
            # If this term word didn't match any sentence word, reject the whole term
            if not found_match:
                return None
        
        return (term, debug_info)
    
    def batch_fuzzy_match(self, terms: List[str], sentence: str) -> Tuple[List[_TextAttr], List[dict]]:
        """
        Match multiple terms against a sentence using subsequence matching.
        Only works for German language.
        
        Args:
            terms: List of candidate terms
            sentence: The sentence to search in
        
        Returns:
            (matches, debug_info)
            matches: List of _TextAttr objects for matched terms
            debug_info: List of detailed match information
        """
        # Early return for non-German languages with correct type
        if self.lang != "de":
            return [], []
        
        matches = []
        debug_data = []
        
        # Pre-filter: remove invalid terms
        valid_terms = [t for t in terms if t and isinstance(t, str) and t.strip()]
        
        if not valid_terms or not sentence:
            return matches, debug_data
        
        # Process each term
        for term in valid_terms:
            result = self.find_fuzzy_matches(term, sentence)
            
            if result:
                matched_term, debug_info = result
                matches.append(_TextAttr(matched_term))
                debug_data.append({
                    'term': matched_term,
                    'details': debug_info
                })
        
        return matches, debug_data
    
    @staticmethod
    def write_debug_log(debug_data: List[dict], output_path: str = "fuzzy_terms.txt"):
        """
        Write fuzzy match debug information to file.
        
        Args:
            debug_data: List of debug dictionaries from batch_fuzzy_match
            output_path: Path to output file
        """
        with open(output_path, "a", encoding="utf-8") as f:
            for entry in debug_data:
                term = entry['term']
                details = entry['details']
                
                parts = []
                for detail in details:
                    part = (f"{detail['term_word']}->{detail['matched_word']}:"
                           f"{detail['matched_subseq']}|"
                           f"missing:{detail['missing']}|"
                           f"extra:{detail['extra']}")
                    parts.append(part)
                
                f.write(f"{term}\t" + " || ".join(parts) + "\n")





# This class finds terms in a sentence

class TermFinder:
    
    # Class-level cache for ngram probabilities (shared across all instances)
    _ngram_probs = None
    _ngram_probs_loaded = False

    def __init__(self, nlp_model, entry_list, raw_entry_list, lang="de", tgt_term_columns=None):
        """
        Initialize the TermMatcher class.

        Args:
            nlp_model: A SpaCy language model instance.
            entry_list: List of entry tuples (processed/lemmatized)
            raw_entry_list: List of entry tuples (raw)
            lang: Language code ('de' or 'it')
            tgt_term_columns: For BISTRO - list of tgt_term column names in order
        """
        self.nlp = nlp_model
        self.entry_list = entry_list
        self.raw_entry_list = raw_entry_list
        self.lang = lang  # Store language explicitly
        self.tgt_term_columns = tgt_term_columns or []  # BISTRO column names
        
        # Load ngram_probs if needed for German and not already loaded
        if self.lang == "de" and not TermFinder._ngram_probs_loaded:
            self._load_ngram_probs()
        
        #define custom fuzzy matcher with language
        self.fuzzy_matcher = FuzzyMatcher(     
            min_word_len=10,                 
            max_diff=3,
            lang=self.lang  # Pass language explicitly
        )
    
    @classmethod
    def _load_ngram_probs(cls):
        """Load ngram probabilities once for all German instances (class-level cache)"""
        NGRAM_PATH = Path.cwd() / "ngram_probs.json"
        
        if not NGRAM_PATH.exists():
            raise FileNotFoundError(f"File not found: {NGRAM_PATH}")
        
        with open(NGRAM_PATH) as f:
            cls._ngram_probs = json.load(f)
        
        cls._ngram_probs_loaded = True
        print("Loaded ngram_probs successfully!")
        

    def check_type(self, terms_list):
        """Check data type and ensure it is List"""
        return isinstance(terms_list, list) and len(terms_list) > 0


    # Longest genuine German compounds run to ~40 chars; anything much longer is
    # almost always degenerate MT output (repetition loops, missing whitespace,
    # glued-together garbage). The n-gram scoring loop below is O(len(word)^2),
    # so without this cap a single pathological "word" can stall the matcher
    # for minutes to hours (this is what causes find_terms.py to appear to hang
    # when scanning many models: more models -> higher odds one row is degenerate).
    MAX_COMPOUND_LEN = 40

    def split_compound(self, word: str) -> List[Tuple[float, str, str]]:
        """Return list of possible splits, best first.
        Only works for German (requires ngram_probs).

        :param word: Word to be split
        :return: List of all splits
        """
        # Early return for non-German
        if self.lang != "de" or not TermFinder._ngram_probs:
            return [(0, word.title(), word.title())]

        # Guard against pathologically long "words" (degenerate MT output) that
        # would otherwise blow up the O(len(word)^2) scoring loop below.
        if len(word) > self.MAX_COMPOUND_LEN:
            return [(0, word.title(), word.title())]

        ngram_probs = TermFinder._ngram_probs  # Use class-level cache

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
                    end_ngram = pre_slice[-k:]  # Look backwards
                    pre_slice_prob.append(ngram_probs["suffix"].get(end_ngram, -1))

                # Probability of ngram in word, if high, split unlikely
                in_ngram = word[n:n+k]
                in_slice_prob.append(ngram_probs["infix"].get(in_ngram, 1))

                # Probability of word starting
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
            pre_slice_prob = max(pre_slice_prob)
            in_slice_prob = min(in_slice_prob)
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


    def _compound_split_matcher(self, sent: str, terms_list: list[str]):
        """
        Compound splitting and matching (German only).
        Returns empty list for non-German languages.
        """
        # Early return for non-German
        if self.lang != "de":
            return []
        
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
    def get_terms_list(self, domain, entry_tuple):
        """
        Extract the appropriate terms list based on domain.
        
        For LegISTyr:
            entry_tuple = (sent, term, other_term_list, other_system_list, [homonym_list])
            domains: "South-Tyrol", "other_tyrol", "other_systems", "homonym"
        
        For BISTRO:
            entry_tuple = (sent, tgt_term_1, tgt_term_2, ...)
            domains: "tgt_term_1", "tgt_term_2", etc. (column names)
        """
        
        # LegISTyr domains
        if domain == "South-Tyrol":
            term = entry_tuple[1]
            # Return None when data is missing (NaN/not a list) so callers can
            # distinguish "no term provided" from "term provided but no match found"
            return list(term) if self.check_type(term) else None

        elif domain == "other_tyrol":
            other_term_list = entry_tuple[2]
            return other_term_list if self.check_type(other_term_list) else None

        elif domain == "other_systems":
            other_system_list = entry_tuple[3]
            return other_system_list if self.check_type(other_system_list) else None

        elif domain == "homonym":
            if len(entry_tuple) > 4:
                homonym_list = entry_tuple[4]
                if self.check_type(homonym_list):
                    term = entry_tuple[1]
                    term_str = term if isinstance(term, str) else str(term)
                    valid_hom = [h for h in homonym_list if h not in term_str]
                    # Still a valid (runnable) list even if empty after filtering
                    return valid_hom
            # No OPTIONS column data at all → skip
            return None

        # BISTRO domains: column names like "tgt_term_1", "tgt_term_2", etc.
        elif domain.startswith("tgt_term_"):
            # Find the index of this column in tgt_term_columns
            if domain in self.tgt_term_columns:
                # Index in tuple: 0=sent, 1=tgt_term_1, 2=tgt_term_2, etc.
                col_index = self.tgt_term_columns.index(domain) + 1  # ← CHANGED from +2 to +1
                
                if col_index < len(entry_tuple):
                    tgt_term_data = entry_tuple[col_index]
                    return tgt_term_data if self.check_type(tgt_term_data) else None
            return None

        else:
            raise Exception(
                f"Invalid domain: {domain}. "
                "For LegISTyr: choose 'South-Tyrol', 'other_tyrol', 'other_systems', or 'homonym'. "
                "For BISTRO: use column names like 'tgt_term_1', 'tgt_term_2', etc."
            )
                
    def find_terms(self, domain):
        """
        Find terms in sentences.

        Args:
            domain: domain to search in 
                   - LegISTyr: "South-Tyrol", "other_tyrol", "other_systems", "homonym"
                   - BISTRO: "tgt_term_1", "tgt_term_2", etc.

        Returns:
            Dictionary mapping sentences (idx, processed_sent) to their matched terms
        """
        results = {}

    # Debug files
        debug_proc = open(f"debug_proc_terms_{domain}.txt", "w", encoding="utf-8")
        debug_raw = open(f"debug_raw_terms_{domain}.txt", "w", encoding="utf-8")

        # Iterate processed and raw entries in parallel.
        # Row-level progress bar so a slow/pathological row is visible
        # immediately instead of the whole model looking frozen (see
        # MAX_COMPOUND_LEN above for why a single row could otherwise stall
        # for minutes).
        rows = list(zip(self.entry_list, self.raw_entry_list))
        for idx, (proc_entry, raw_entry) in enumerate(
            tqdm(rows, desc=f"Rows [{domain}]", unit="row", leave=False)
        ):
            
            sent_id = (idx, proc_entry[0])  # (index, sentence)

            # Ensure strings
            sent_str = proc_entry[0] if isinstance(proc_entry[0], str) else ""
            raw_sent_str = raw_entry[0] if isinstance(raw_entry[0], str) else ""

            # Get the terms lists for this domain.
            # None means there was no term to be matched → matcher should be skipped.
            # [] means term existed but yielded no valid match after filtering.
            terms_list = self.get_terms_list(domain, proc_entry)
            raw_terms_list = self.get_terms_list(domain, raw_entry)

            # Write to debug files
            debug_proc.write(f"{idx}: {terms_list}\n")
            debug_raw.write(f"{idx}: {raw_terms_list}\n")

            # If BOTH term lists are None the row had no term annotation at all.
            # Store None so success rate calculation can exclude it from the denominator.
            if raw_terms_list is None:
                results[sent_id] = None
                continue

            # From here on, treat None the same as [] for matching purposes. Otherwise the term matcher breaks
            # (to stay on the safe side, we make sure both term lists are empty. This check is actually redundant).
            raw_terms_list = raw_terms_list if raw_terms_list is not None else []
            terms_list = terms_list if terms_list is not None else []
 
            # Now actually match terms
            matches = []
 
            # Match 1: raw terms and sentence
            if raw_terms_list and raw_sent_str:
                pattern_match = self.phrase_matcher(raw_sent_str, raw_terms_list)
 
                matches = [
                    m for m in pattern_match
                    if m and str(m).strip() and str(m).lower() != "nan"
                ]
 
            # Match 2: Try matching with lemmatized terms and sentence
            if not matches and terms_list and sent_str:
                pattern_match = self.phrase_matcher(sent_str, terms_list)
                
                # Match 3: Compound splitting (German only)
                if not pattern_match:
                    pattern_match = self._compound_split_matcher(sent_str, terms_list)
 
                matches = [
                    m for m in pattern_match
                    if m and str(m).strip() and str(m).lower() != "nan"
                ]
 
            # Match 4: Fuzzy matching for inflectional variants
            if not matches and raw_terms_list and raw_sent_str:
                fuzzy_matches, debug_fuzzy = self.fuzzy_matcher.batch_fuzzy_match(
                    raw_terms_list, 
                    raw_sent_str
                )
                
                if fuzzy_matches:
                    self.fuzzy_matcher.write_debug_log(debug_fuzzy, "fuzzy_terms.txt")
                    matches = fuzzy_matches
 
            results[sent_id] = matches
 
        return results