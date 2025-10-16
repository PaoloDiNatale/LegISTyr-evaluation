from configparser import ConfigParser
import argparse

import pandas as pd
import spacy

from spacy.lang.de import German
from spacy.matcher import PhraseMatcher


from utils.term_finder_utils import create_entries, TermFinder
from utils.results_utils import save_term_results, find_terms_over_models, print_success_rate

config = ConfigParser()
config.read('config.ini')

nlp_de = German()
matcher_de = PhraseMatcher(nlp_de.vocab, attr="LOWER")

#Parsing arguments

parser = argparse.ArgumentParser(description='Create entries from translation table')

parser.add_argument('--hom', action="store_true",
                       help='Include if you are testing on the honomym subset')
    
args = parser.parse_args()


#import data
if args.hom:
    df = pd.read_csv('data/preprocessed_data_homs.csv', delimiter=';', encoding='utf-8-sig')
else:
    df = pd.read_csv('data/preprocessed_data_2.csv', delimiter=';', encoding='utf-8-sig')


models_str = config.get('main', 'models')
models_list = [item.strip() for item in models_str.split(',')]

print(models_list)

entries_dict = create_entries(df, models_list, homonym=args.hom)

#CHECK WHAT WE HAVE
# After creating entries_dict, check what you have:
#print("\n=== CHECKING ENTRY DATA ===")
#for col in models_list:
#    entry_list = entries_dict[col]
#    print(f"\n{col}: {len(entry_list)} entries")
    
    # Check first few entries
#    for i, (sent, term, other_term, other_sys, *homonym) in enumerate(entry_list[:4]):
#        print(f"  Entry {i}:")
#        print(f"    Sentence: {sent[5:8]}...")
#        print(f"    Term: {term}")
#        print(f"    Type of term: {type(term)}")



#TO DO: optimize this iteration
#Now find terms

# Find terms in the sentences. Returns a dictionary where the key is the model name, the values is a dict {"sentence": [term_matches]}
st_term_results = find_terms_over_models(nlp_de, entries_dict, models_list, "South-Tyrol")
#print("HAVE A LOOK HERE")
#print(type(st_term_results))
#print(st_term_results)
other_st_term_results = find_terms_over_models(nlp_de, entries_dict, models_list, "other_tyrol")
other_legal_system_results = find_terms_over_models(nlp_de, entries_dict, models_list, "other_systems")
if args.hom:
    wrong_homonym_results = find_terms_over_models(nlp_de, entries_dict, models_list, "homonym")


### SAVE AS CSV TO VISUALIZE RESULTS
get_st_results = save_term_results(st_term_results, filename="South_Tyrol_terms")
get_other_st_results = save_term_results(other_st_term_results, filename="other_south_tyrol_terms")
get_other_legal_system_results = save_term_results(other_legal_system_results, filename="other_legal_systems_terms")
if args.hom:
    get_other_legal_system_results = save_term_results(wrong_homonym_results, filename="wrong_homonyms")


###SAVE RESULTS OF SUCCESS RATE
print_success_rate(
    st_term_results,
    category_name="Success rate of target South-Tyrolean terms",
    clear_file=True
)

print_success_rate(
    other_st_term_results,
    category_name="Success rate of alternative South-Tyrolean terms"
)

print_success_rate(
    other_legal_system_results,
    category_name="Success rate of terms from extraneous legal systems"
)

if args.hom:
    print_success_rate(
        wrong_homonym_results,
        category_name="Percentage of incorrect homonym insertion"
    )