### Create environment

\`\`\`
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

\`\`\`

### Load spacy models according to target language

\`\`\`
python -m spacy download de_core_news_sm
\`\`\`

### Preprocess files

#### Put your translations in txt format, one instance per line, in the correspodning data folder. You will find the testset data already there as a csv file.

#### The '--hom' flag is used when working on the homonym subset. If omitted, it defaults to simple term finder

#### Lang can be either de or it, defaults to de

\`\`\`
python preproc_2.py --hom --lang de
\`\`\`

#### This will create 'preprocessed_data.csv" file, with your translations appended. In confif.ini file, you will find the name of the models.

### Match terms

#### Run:

\`\`\`
python find_terms_2.py --hom --lang de
\`\`\`

### Lang flag still doesn't do anything, I will fix it when I can. 
### You find the matches in results folder and the final percentages in data/results_analysis.