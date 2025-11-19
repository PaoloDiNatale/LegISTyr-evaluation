from configparser import ConfigParser
import argparse

import pandas as pd
import spacy

from spacy.lang.de import German
from spacy.lang.it import Italian
from spacy.matcher import PhraseMatcher

#reading model names
config = ConfigParser()
config.read('config.ini')


#Parsing arguments
parser = argparse.ArgumentParser(description='Choose challenge sets and target language.')

parser.add_argument('--hom', action="store_true",
                       help='Include if you are testing on the honomym subset')

parser.add_argument('--lang', choices=['de', 'it'], default='de',
    help='Choose your target language: "de" (Deutsch) or "it" (Italian). Default is "de".'
)

args = parser.parse_args()

#Load model and matcher
if args.lang == 'de':
    nlp_lang = German()
    print("German model and matcher loaded.")
elif args.lang == 'it':   
    nlp_lang = Italian()
    print("Italian model and matcher loaded.")
else:
    raise ValueError("Unsupported language. Please choose 'de' or 'it'.")

#Setting lang argument as global
from utils.config.config import set_lang
set_lang(args.lang)

#Finally importing utils that depend on the language setting
from utils.term_finder_utils import create_entries, TermFinder
from utils.results_utils import save_term_results, find_terms_over_models, print_success_rate


df = pd.read_csv('data/preprocessed_data_2.csv', delimiter=';', encoding='utf-8-sig')


# You should run the return_spans function on just the source column and return the spans (as list i think)
