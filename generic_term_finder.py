#call example: python generic_term_finder.py   --corpus-txt data_generic/lemmatized_corpus.txt   --terms-csv data/homonyms/LegISTyr__homonyms.csv   --term-column "TARGET HYPOTHESIS (DE SOUTH TYROL)" --output data_generic/count_homonyms.csv

import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from spacy.lang.de import German
from spacy.matcher import PhraseMatcher


def load_corpus_txt(path: Path, encoding: str = "utf-8") -> str:
    if path.suffix.lower() != ".txt":
        raise ValueError("Corpus must be a .txt file")
    return path.read_text(encoding=encoding)


def load_corpus_lines(path: Path, encoding: str = "utf-8") -> list[str]:
    """Load corpus as a list of lines (one sentence per line)."""
    if path.suffix.lower() != ".txt":
        raise ValueError("Corpus must be a .txt file")
    return path.read_text(encoding=encoding).splitlines()


def parse_term_cell(val: str) -> list[str]:
    """
    Parse a term cell that is either:
      - a plain string:                  'Invalidität'
      - a bracketed numpy-style list:    "['elektrisches Handwerkzeug' 'tragbares Elektrowerkzeug']"

    Returns a list of individual term strings.
    """
    val = val.strip()

    if val.startswith("[") and val.endswith("]"):
        terms = re.findall(r"'([^']+)'", val)
        return [t.strip() for t in terms if t.strip()]
    else:
        return [val] if val else []


def load_terms_from_csv(
    path: Path, term_column: str, encoding: str = "utf-8"
) -> tuple[list[str], list[list[str]], list[str]]:
    """
    Returns:
      - flat_terms:   deduplicated flat list of all individual terms (for matcher)
      - row_terms:    list of term lists, one per source row (preserves row structure)
      - concept_ids:  list of concept IDs, one per source row;
                      uses 'concept_id' column if present, otherwise row index
    """
    ext = path.suffix.lower()
    if ext == ".csv":
        sep = ";"
    elif ext == ".tsv":
        sep = "\t"
    else:
        raise ValueError("Terms file must be a .csv or .tsv file")

    df = pd.read_csv(path, encoding=encoding, sep=sep)

    if term_column not in df.columns:
        raise ValueError(f'Terms file does not contain column "{term_column}".')

    has_concept_id = "concept_id" in df.columns

    row_terms = []
    all_terms_flat = []
    concept_ids = []

    for i, row in df.iterrows():
        val = row[term_column]
        if pd.isna(val):
            continue
        parsed = parse_term_cell(str(val))
        row_terms.append(parsed)
        all_terms_flat.extend(parsed)
        raw_id = row["concept_id"] if has_concept_id else i
        try:
            concept_ids.append(int(raw_id))
        except (ValueError, TypeError):
            concept_ids.append(int(i))  # fallback to row index if cast fails

    # deduplicate flat list while preserving order
    flat_terms = list(dict.fromkeys(t for t in all_terms_flat if t))

    return flat_terms, row_terms, concept_ids


def build_matcher(nlp_lang, terms: list[str]) -> PhraseMatcher:
    matcher = PhraseMatcher(nlp_lang.vocab, attr="LOWER")
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
    term_to_id = {t: nlp_lang.vocab.strings[t] for t in all_terms}
    id_counts = Counter({mid: 0 for mid in term_to_id.values()})

    for chunk in iter_text_chunks(corpus, chunk_size=chunk_size, overlap=overlap):
        doc = nlp_lang(chunk)
        for match_id, _, _ in matcher(doc):
            id_counts[match_id] += 1

    return {term: int(id_counts[mid]) for term, mid in term_to_id.items()}


def match_terms_per_sentence(
    lines: list[str],
    nlp_lang,
    matcher,
    term_to_concept: dict[str, str],
) -> list[list[str]]:
    """
    For each sentence line, return the deduplicated list of concept IDs
    whose terms matched in that line.

    Args:
        lines:           list of corpus sentences (one per line)
        nlp_lang:        spaCy language model
        matcher:         PhraseMatcher built from all terms
        term_to_concept: mapping from term string → concept_id

    Returns:
        list of lists — one per line, each a sorted list of matched concept IDs
    """
    # reverse-map match_id (vocab int) → concept_id
    id_to_concept: dict[int, int] = {
        nlp_lang.vocab.strings[term]: concept_id
        for term, concept_id in term_to_concept.items()
    }

    print(f"  term_to_concept size: {len(term_to_concept)}")
    print(f"  id_to_concept size:   {len(id_to_concept)}")
    print(f"  sample terms: {list(term_to_concept.items())[:3]}")

    # test matcher on first non-empty line
    for line in lines:
        if line.strip():
            doc = nlp_lang(line)
            matches = list(matcher(doc))
            print(f"  test line:          {line[:80]}")
            print(f"  raw matcher output: {matches[:5]}")
            break

    results = []
    diagnostic_lines = []

    for i, line in enumerate(lines):
        if not line.strip():
            results.append([])
            diagnostic_lines.append(f"{i}\t[]\t[]\n")
            continue
        doc = nlp_lang(line)
        matched_concepts: set[int] = set()
        matched_terms: list[str] = []
        for match_id, start, end in matcher(doc):
            concept_id = id_to_concept.get(match_id)
            if concept_id is not None:
                matched_concepts.add(concept_id)
                matched_terms.append(doc[start:end].text)
        sorted_concepts = sorted(matched_concepts)
        results.append(sorted_concepts)
        diagnostic_lines.append(
            f"{i}\t{sorted_concepts}\t{matched_terms}\n"
        )

    # Write diagnostic file alongside the output
    with open("diagnostic_per_sentence.txt", "w", encoding="utf-8") as f:
        f.write("sentence_idx\tmatched_concept_ids\tmatched_terms\n")
        f.writelines(diagnostic_lines)

    return results


def build_term_to_concept(
    row_terms: list[list[str]], concept_ids: list[int]
) -> dict[str, int]:
    """Map every individual term to its concept ID (integer)."""
    mapping = {}
    for terms, concept_id in zip(row_terms, concept_ids):
        for term in terms:
            mapping[term] = concept_id
    return mapping


def save_results(
    term_freq: dict[str, int],
    row_terms: list[list[str]],
    concept_ids: list[str],
    output_path: Path,
    output_format: str = "per_term",
    sort: str = None,
    encoding: str = "utf-8",
    # per_sentence only
    sentence_matches: list[list[str]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()

    if output_format == "per_term":
        # ── One row per individual term ──────────────────────────────────────
        if ext == ".json":
            data = term_freq
            if sort == "frequency":
                data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
            elif sort == "term":
                data = dict(sorted(data.items()))
            output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding=encoding)

        elif ext == ".csv":
            df = pd.DataFrame({"term": list(term_freq.keys()), "frequency": list(term_freq.values())})
            if sort == "frequency":
                df.sort_values("frequency", ascending=False, inplace=True)
            elif sort == "term":
                df.sort_values("term", inplace=True)
            df.to_csv(output_path, index=False, encoding=encoding)

        else:
            raise ValueError("Output must be .json or .csv")

    elif output_format == "per_concept":
        # ── One row per source row: term list + summed total ─────────────────
        rows = []
        for terms in row_terms:
            total = sum(term_freq.get(term, 0) for term in terms)
            rows.append({"terms": ", ".join(terms), "total": total})

        df = pd.DataFrame(rows)

        if ext == ".csv":
            df.to_csv(output_path, index=False, encoding=encoding)
        elif ext == ".json":
            output_path.write_text(
                json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2),
                encoding=encoding
            )
        else:
            raise ValueError("Output must be .json or .csv")

    elif output_format == "per_sentence":
        # ── One row per corpus sentence: sentence_idx + matched concept IDs ──
        rows = [
            {"sentence_idx": i, "matched_concept_ids": json.dumps(matched)}
            for i, matched in enumerate(sentence_matches)
        ]
        df = pd.DataFrame(rows)

        if ext == ".csv":
            df.to_csv(output_path, index=False, encoding=encoding)
        elif ext == ".json":
            output_path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2),
                encoding=encoding
            )
        else:
            raise ValueError("Output must be .json or .csv")

    else:
        raise ValueError(
            f"Unknown output format: {output_format}. "
            "Choose 'per_term', 'per_concept', or 'per_sentence'."
        )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PhraseMatcher term frequency counter for German corpus")
    p.add_argument("-c", "--corpus-txt", required=True, help="Path to corpus .txt (one sentence per line)")
    p.add_argument("-t", "--terms-csv", required=True, help="Path to terms file (.csv or .tsv)")
    p.add_argument("--term-column", required=True, help="Column name in terms file containing the terms")
    p.add_argument("-o", "--output", required=True, help="Output path (.json or .csv)")
    p.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    p.add_argument(
        "--sort",
        choices=["frequency", "term"],
        default=None,
        help="Sort output by 'frequency' (descending) or 'term' (alphabetical). Only applies to per_term format. Default: preserve input order."
    )
    p.add_argument(
        "--output-format",
        choices=["per_term", "per_concept", "per_sentence"],
        default="per_term",
        help=(
            "per_term:     one row per individual term with its corpus frequency (default). "
            "per_concept:      one row per source row with summed term counts. "
            "per_sentence: one row per corpus sentence with matched concept IDs."
        )
    )
    return p


def main():
    args = build_argparser().parse_args()

    nlp_lang = German()
    print("German model loaded for tokenization and matching.")
    print(f"Loading terms from {args.terms_csv}...")

    flat_terms, row_terms, concept_ids = load_terms_from_csv(
        Path(args.terms_csv), term_column=args.term_column, encoding=args.encoding
    )
    print(f"Loaded {len(flat_terms)} unique terms across {len(row_terms)} source rows.")
    print("Building matcher...")
    matcher = build_matcher(nlp_lang, flat_terms)

    sentence_matches = None

    if args.output_format == "per_sentence":
        # ── sentence mode: load corpus as lines, match per line ──────────────
        print(f"Loading corpus lines from {args.corpus_txt}...")
        lines = load_corpus_lines(Path(args.corpus_txt), encoding=args.encoding)
        print(f"Loaded {len(lines)} corpus lines. Matching...")

        term_to_concept = build_term_to_concept(row_terms, concept_ids)
        sentence_matches = match_terms_per_sentence(lines, nlp_lang, matcher, term_to_concept)
        term_freq = {}  # not needed for this mode

    else:
        # ── chunk mode: load corpus as one string, count frequencies ─────────
        print(f"Loading corpus from {args.corpus_txt}...")
        corpus = load_corpus_txt(Path(args.corpus_txt), encoding=args.encoding)
        print("Matching terms in corpus...")
        term_freq = count_terms_over_chunks(
            corpus=corpus,
            nlp_lang=nlp_lang,
            matcher=matcher,
            all_terms=flat_terms,
            chunk_size=500_000,
            overlap=0,
        )

    save_results(
        term_freq=term_freq,
        row_terms=row_terms,
        concept_ids=concept_ids,
        output_path=Path(args.output),
        output_format=args.output_format,
        sort=args.sort,
        encoding=args.encoding,
        sentence_matches=sentence_matches,
    )
    print(f"✅ Results saved to: {args.output}")


if __name__ == "__main__":
    print("Starting execution")
    main()