
#call example: python generic_term_finder.py   --corpus-txt data_generic/lemmatized_corpus.txt   --terms-csv data/homonyms/LegISTyr__homonyms.csv   --term-column "TARGET HYPOTHESIS (DE SOUTH TYROL)" --output data_generic/count_homonyms.csv

import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd
from spacy.lang.de import German
from spacy.matcher import PhraseMatcher


def load_corpus_txt(path: Path, encoding: str = "utf-8") -> str:
    if path.suffix.lower() != ".txt":
        raise ValueError("Corpus must be a .txt file")
    return path.read_text(encoding=encoding)


def load_terms_from_csv(path: Path, term_column: str, encoding: str = "utf-8") -> list[str]:
    if path.suffix.lower() != ".csv":
        raise ValueError("Terms file must be a .csv file")

    df = pd.read_csv(path, encoding=encoding, sep=";")

    if term_column not in df.columns:
        raise ValueError(f'Terms CSV does not contain column "{term_column}".')

    terms = (
        df[term_column]
        .dropna()
        .astype(str)
        .map(str.strip)
        .tolist()
    )

    # deduplicate while preserving order
    return list(dict.fromkeys(t for t in terms if t))


def build_matcher(nlp_lang, terms: list[str]) -> PhraseMatcher:
    matcher = PhraseMatcher(nlp_lang.vocab, attr="LOWER")  # case-insensitive matching

    # one pattern per rule label (the label IS the term)
    for term in terms:
        pattern = nlp_lang.make_doc(term)
        matcher.add(term, [pattern])

    return matcher


def iter_text_chunks(text: str, chunk_size: int = 500_000, overlap: int = 2000):
    i = 0
    n = len(text)
    step = max(1, chunk_size - overlap)
    while i < n:
        yield text[i:i + chunk_size]
        i += step


def count_terms_over_chunks(
    corpus: str,
    nlp_lang,
    matcher,
    all_terms: list[str],
    chunk_size: int = 500_000,
    overlap: int = 2000,
) -> dict[str, int]:
    # map term -> match_id once
    term_to_id = {t: nlp_lang.vocab.strings[t] for t in all_terms}

    # initialize all to zero
    id_counts = Counter({mid: 0 for mid in term_to_id.values()})

    for chunk in iter_text_chunks(corpus, chunk_size=chunk_size, overlap=overlap):
        doc = nlp_lang(chunk)
        for match_id, _, _ in matcher(doc):
            id_counts[match_id] += 1

    # convert back to term -> freq (ensures all terms exist)
    return {term: int(id_counts[mid]) for term, mid in term_to_id.items()}


def save_results(term_freq: dict[str, int], output_path: Path, encoding: str = "utf-8") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ext = output_path.suffix.lower()
    if ext == ".json":
        output_path.write_text(json.dumps(term_freq, ensure_ascii=False, indent=2), encoding=encoding)
    elif ext == ".csv":
        df = pd.DataFrame({"term": list(term_freq.keys()), "frequency": list(term_freq.values())})
        df.sort_values("frequency", ascending=False, inplace=True)
        df.to_csv(output_path, index=False, encoding=encoding)
    else:
        raise ValueError("Output must be .json or .csv")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PhraseMatcher term frequency counter for German corpus")
    p.add_argument("-c", "--corpus-txt", required=True, help="Path to lemmatized corpus .txt")
    p.add_argument("-t", "--terms-csv", required=True, help="Path to terms .csv")
    p.add_argument("--term-column", required=True, help="Column name in terms CSV containing the terms")
    p.add_argument("-o", "--output", required=True, help="Output path (.json or .csv)")
    p.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    return p


def main():
    args = build_argparser().parse_args()

    nlp_lang = German()  # tokenizer + vocab is enough for PhraseMatcher
    print("German model loaded for tokenization and matching.")
    print(f"Loading corpus from {args.corpus_txt} and terms from {args.terms_csv}...")

    corpus = load_corpus_txt(Path(args.corpus_txt), encoding=args.encoding)
    terms = load_terms_from_csv(Path(args.terms_csv), term_column=args.term_column, encoding=args.encoding)
    print(f"Loaded the following term: {terms}")
    print("Building matcher...")
    matcher = build_matcher(nlp_lang, terms)

    print("Matching terms in corpus...")
    term_freq = count_terms_over_chunks(
        corpus=corpus,
        nlp_lang=nlp_lang,
        matcher=matcher,
        all_terms=terms,
        chunk_size=500_000,
        overlap=0,
    )

    save_results(term_freq, Path(args.output), encoding=args.encoding)


if __name__ == "__main__":
    print("Starting execution")
    main()

