#how to call me:
#python generic_preproc.py   -i data_generic/Train_BTCatex_de.txt   -o data_generic/lemmatized_corpus.txt

import argparse
import time
from pathlib import Path

import pandas as pd
import spacy

from utils.preproc_utils import batch_lemmatize_sentences


def load_sentences(
    input_file_path: Path,
    column_name: str,
    sep: str,
    encoding: str,
):
    ext = input_file_path.suffix.lower().lstrip(".")

    if ext in ("csv", "tsv"):
        effective_sep = sep if ext == "tsv" else ","
        df = pd.read_csv(input_file_path, sep=effective_sep, encoding=encoding)

        if column_name not in df.columns:
            raise ValueError(f'Dataframe does not contain column "{column_name}".')

        sentences = (
            df[column_name]
            .dropna()
            .astype(str)
            .map(str.strip)
            .tolist()
        )

    elif ext == "txt":
        with open(input_file_path, "r", encoding=encoding) as f:
            sentences = [line.strip() for line in f if line.strip()]

    else:
        raise ValueError("Input must be .csv, .tsv, or .txt")

    return sentences  # Limit to first 1000 sentences for testing


def tsv_column_to_single_lemmatized_txt(
    input_file_path: str,
    output_file_path: str,
    column_name: str = "filled_text",
    sep: str = "\t",
    encoding: str = "utf-8",
    batch_size: int = 100,
) -> None:
    print("Loading spaCy model...", end=" ")
    model = spacy.load("de_dep_news_trf")
    print("Done!")

    input_path = Path(input_file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("Loading data...", end=" ")
    sentences = load_sentences(
        input_file_path=input_path,
        column_name=column_name,
        sep=sep,
        encoding=encoding,
    )
    print(f"Loaded {len(sentences)} sentences!")

    print("\nLemmatizing and joining...", end="\n")
    a = time.time()

    lemmatized_rows = batch_lemmatize_sentences(
        sentences=sentences,
        model=model,
        batch_size=batch_size,
    )

    #insert newlines between sentences if needed
    joined_text = " ".join(lemmatized_rows)

    b = time.time() - a
    print("Done!")
    print(f"Execution time:\t{b/60:.2f} min.")

    print("\nSaving the data... ", end="")
    out_path = Path(output_file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding=encoding) as f:
        f.write(joined_text)
    print("Done!")
    print(f"Data saved in {out_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lemmatize text from CSV/TSV column or TXT file and save as a single TXT"
    )
    p.add_argument(
        "-i", "--input-file",
        required=True,
        help="Path to input .csv, .tsv, or .txt file"
    )
    p.add_argument(
        "-o", "--output-file",
        required=True,
        help="Path to output .txt file"
    )
    p.add_argument(
        "-c", "--column-name",
        default="filled_text",
        help='Column name for CSV/TSV input (ignored for .txt)'
    )
    return p


def main():
    args = build_argparser().parse_args()

    tsv_column_to_single_lemmatized_txt(
        input_file_path=args.input_file,
        output_file_path=args.output_file,
        column_name=args.column_name,
    )


if __name__ == "__main__":
    main()


#apply tqdm: tqdm(nlp.pipe(sentences), total=len(sentences))