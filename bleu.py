# python ./bleu.py --testset legistyr --mode hom --metric chrf     --hypotheses_dir /home/pdinatale/term_finder/LegISTyr-evaluation/data/legistyr/hom     --reference /home/pdinatale/term_finder/LegISTyr-evaluation/data/legistyr/hom/references_hom.txt
import argparse
import os
from pathlib import Path

import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(
    description='Compute BLEU or chrF scores for MT outputs and attach them to the rates TSV.'
)

parser.add_argument('--testset',
                    type=str,
                    required=True,
                    choices=['legistyr', 'bistro'],
                    help='Test set to use: legistyr or bistro')

parser.add_argument('--mode',
                    type=str,
                    required=True,
                    help='Mode: for legistyr [hom, abbr, simple_terms], for bistro [var, rel, hom]')


parser.add_argument('--metric',
                    type=str,
                    required=True,
                    choices=['bleu', 'chrf'],
                    help='MT metric to compute: bleu or chrf')

parser.add_argument('--hypotheses_dir',
                    type=str,
                    required=True,
                    help='Directory containing one .txt file per model (filename = model name, one sentence per line)')

parser.add_argument('--reference',
                    type=str,
                    required=True,
                    help='Path to reference file (one sentence per line, aligned with hypotheses)')

args = parser.parse_args()

print(f"Testset:        {args.testset}")
print(f"Mode:           {args.mode}")
print(f"Metric:         {args.metric}")
print(f"Hypotheses dir: {args.hypotheses_dir}")
print(f"Reference:      {args.reference}")

# ============================================================================
# VALIDATE MODE FOR TESTSET
# ============================================================================

if args.testset == 'legistyr':
    valid_modes = ['hom', 'abbr', 'simple_terms']
    if args.mode not in valid_modes:
        raise ValueError(f"Invalid mode '{args.mode}' for testset 'legistyr'. Choose from: {valid_modes}")
elif args.testset == 'bistro':
    valid_modes = ['var', 'rel', 'hom']
    if args.mode not in valid_modes:
        raise ValueError(f"Invalid mode '{args.mode}' for testset 'bistro'. Choose from: {valid_modes}")

# ============================================================================
# LOAD REFERENCE
# ============================================================================

reference_path = Path(args.reference)
if not reference_path.exists():
    raise FileNotFoundError(f"Reference file not found: {reference_path}")

references_df = pd.read_csv(reference_path, sep='\t', encoding='utf-8')
if 'context' not in references_df.columns:
    raise ValueError(f"Column 'context' not found in {reference_path}. Available columns: {list(references_df.columns)}")
references = references_df['context'].astype(str).tolist()

# ============================================================================
# LOCATE TSV TO ATTACH RESULTS TO
# ============================================================================

base_dir = Path('./data')
analysis_dir = base_dir / 'results_analysis' / args.testset / args.mode
tsv_path = analysis_dir / 'term_accuracy_rates.tsv'

if not tsv_path.exists():
    raise FileNotFoundError(
        f"Rates TSV not found: {tsv_path}\n"
        f"Run find_terms.py first to generate it."
    )

results_df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')
print(f"Rates TSV loaded: {tsv_path}  ({len(results_df)} models)")

# ============================================================================
# INITIALISE METRIC
# ============================================================================

if args.metric == 'bleu':
    scorer = BLEU(tokenize='13a')   # standard tokeniser; use 'char' for char-level
elif args.metric == 'chrf':
    scorer = CHRF(word_order=2)

metric_col = args.metric   # column name in the TSV

# ============================================================================
# SCORE EACH MODEL
# ============================================================================

hypotheses_dir = Path(args.hypotheses_dir)
if not hypotheses_dir.is_dir():
    raise NotADirectoryError(f"Hypotheses directory not found: {hypotheses_dir}")

scores = {}   # model_name → score (float)

#define names of the models
hypothesis_files = sorted(hypotheses_dir.glob('*.txt'))
if not hypothesis_files:
    raise FileNotFoundError(f"No .txt files found in: {hypotheses_dir}")

print(f"\n{'='*60}")
print(f"Scoring with {args.metric.upper()}")
print(f"{'='*60}")

for hyp_file in hypothesis_files:
    model_name = hyp_file.stem   # filename without extension = model name

    with open(hyp_file, encoding='utf-8') as f:
        hypotheses = [line.rstrip('\n') for line in f]

    if len(hypotheses) != len(references):
        raise ValueError(
            f"Length mismatch for '{model_name}': "
            f"{len(hypotheses)} hypotheses vs {len(references)} references"
        )

    result = scorer.corpus_score(hypotheses, [references], n_bootstrap=1000)

    if args.metric == 'bleu':
        score = round(result.score, 2)           # BLEU score (0–100)
        signature = result.format()
    elif args.metric == 'chrf':
        score = round(result.score, 4)           # chrF score (0–100)
        signature = result.format()

    scores[model_name] = score
    print(f"  {model_name}:")
    print(f"Result details:    {result}")
    print(f"Signature:    {scorer.get_signature()}")

# ============================================================================
# ATTACH SCORES TO TSV
# ============================================================================

# Map scores onto the TSV by model name, overwriting the column if it already
# exists (re-run). Using .map() on the model column avoids any join ambiguity.
results_df[metric_col] = results_df['model'].map(scores)

results_df.to_csv(tsv_path, sep='\t', index=False, encoding='utf-8')

print(f"\n✅ {args.metric.upper()} scores attached to: {tsv_path}")

# Warn about any models in the TSV that had no matching hypothesis file
missing = results_df.loc[results_df[metric_col].isna(), 'model'].tolist()
if missing:
    print(f"  ⚠️  No hypothesis file found for: {missing} — these rows have NaN in '{metric_col}'")