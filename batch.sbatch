#!/bin/bash
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=gpu-pre
#SBATCH --account=eurac-commul
#SBATCH --time=06:00:00
#SBATCH --output=matcher-%j.out
#SBATCH --error=matcher-%j.err
#SBATCH --gres=gpu:1
###SBATCH --gres=gpu:a100-sxm4-80gb:1
###SBATCH --constraint=gpu80g

module load python/3.10.8-gcc-12.1.0-linux-ubuntu22.04-zen2
module load cudnn/8.9.7.29-12-gcc-12.1.0
 
##python -m venv .venv
source .venv/bin/activate

#pip install sacrebleu >= 2.3.1
#pip install cupy-cuda12x

python -c "import cupy; print(cupy.__version__); print(cupy.cuda.runtime.getDeviceCount())"

#pip freeze > requirements.txt


python ./preproc.py --testset bistro --mode rel --lang de
python ./find_terms.py --testset bistro --mode rel --lang de
python ./bleu.py --testset bistro --mode rel --metric chrf     --hypotheses_dir ./data/bistro/rel     --reference ./data/bistro/rel/BISTRO__rel.tsv
python ./bleu.py --testset bistro --mode rel --metric bleu     --hypotheses_dir ./data/bistro/rel     --reference ./data/bistro/rel/BISTRO__rel.tsv

#python generic_term_finder.py  --output-format per_row --corpus-txt data_generic/lemmatized_corpus.txt   --terms-csv ./data/bistro/var/BISTRO__var.tsv   --term-column "tgt_term_gold" --output data_generic/count_var.csv