from typing import List
from math import ceil
from unicodedata import normalize
import tempfile

import argparse
import pandas as pd
import nltk
import hazm

from tqdm.auto import tqdm
from os import path, makedirs

import sys
import os

os.environ["LASER"] = "/homes/julez/ts-adapters/src/data/cochrane/LASER"

sys.path.append("/homes/julez/ts-adapters/src/data/cochrane/")
sys.path.append("/homes/julez/ts-adapters/src/data/cochrane/vecalign")
sys.path.append("/homes/julez/ts-adapters/src/data/cochrane/LASER/source/")
from vecalign import overlap, vecalign, dp_utils
from LASER.source import embed

LANGUAGES = ["en", "es", "fr", "fa"]

SENTENCE_SPLITTER = {
    "en": lambda x: nltk.sent_tokenize(x, language="english"),
    "es": lambda x: nltk.sent_tokenize(x, language="spanish"),
    "fr": lambda x: nltk.sent_tokenize(x, language="french"),
    "fa": lambda x: hazm.sent_tokenize(x)
}


def normalize_content(content: dict):
    """Dictionary with
    {
        "language": {
            "title": "...",
            "abstract": [{"heading", "text"}],
            "pls_title",
            "pls_type",
            "pls",
        }
    }

    Return as:

    {
        "en_abstract": ["sent1", "sent2", ...],
        "es_abstract": ["sent1", "sent2", ...],
        ...
    }
    """

    normalized = {}
    for lang, data in content.items():
        if lang not in LANGUAGES:
            continue
        sent_tokenize = SENTENCE_SPLITTER[lang]

        abstract = data["abstract"]
        abstract_flat = []
        for paragraph in abstract:
            sents = sent_tokenize(paragraph["text"])
            for sent in sents:
                abstract_flat.append(sent)
        normalized[f"{lang}_abstract"] = abstract_flat

    return normalized


def main(align_lang):
    print("#" * 50)
    print("MARKER 1")
    df_scraped = pd.read_json("/homes/julez/ts-adapters/src/data/cochrane/scraped_data/data.json")
    df_scraped = pd.concat([
        df_scraped,
        pd.json_normalize(df_scraped["content"].apply(normalize_content))
    ], axis=1)
    df_scraped = df_scraped.set_index("doi")
    # df_scraped.head()

    print("#" * 50)
    print("MARKER 2")
    frames = []
    for lang in ["en", "fr", "es", "fa"]:
        for split in ["train", "val", "test"]:
            df = pd.read_csv(f"/homes/julez/datasets-raw/cochrane_sent/unfiltered (r=0)/{lang}/{split}0_{lang}.csv",
                             index_col=0)
            df["lang"] = lang
            df["split"] = split
            frames.append(df)

    df_alignments = pd.concat(frames)
    df_alignments.head()

    if lang == "fa":
        print("-" * 50)
        print("FARSI")
        ENCODER_MODEL = embed.SentenceEncoder(
            model_path="/homes/julez/ts-adapters/src/data/cochrane/LASER/laser3-pes_Arab.v1.pt",
            spm_vocab="/homes/julez/ts-adapters/src/data/cochrane/LASER/laser3-pes_Arab.v1.cvocab",
        )
    else:
        ENCODER_MODEL = embed.SentenceEncoder(model_path="/homes/julez/ts-adapters/src/data/cochrane/LASER/laser2.pt")

    # Vecalign parameters
    alignment_max_size = 1
    one_to_many = 1
    search_buffer_size = 5
    del_percentile_frac = 0.2
    max_size_full_dp = 300
    costs_sample_size = 20000
    num_samps_for_norm = 100
    print_aligned_text = False

    def write_sents(sents: List[str], out_file):
        sents = [s.replace("\n", "") for s in sents]
        with open(out_file, "w") as fout:
            fout.writelines("\n".join(sents))

    def align(src: List[str], tgt: List[str], src_lang, tgt_lang):
        print("#" * 50)
        print("MARKER 3")
        with tempfile.TemporaryDirectory() as tmpdirname:
            src_file = os.path.join(tmpdirname, f"src.{src_lang}")
            tgt_file = os.path.join(tmpdirname, f"tgt.{tgt_lang}")
            src_emb = os.path.join(tmpdirname, f"src.{src_lang}.emb")
            tgt_emb = os.path.join(tmpdirname, f"tgt.{tgt_lang}.emb")

            write_sents(src, src_file)
            write_sents(tgt, tgt_file)

            embed.embed_sentences(
                ifname=src_file,
                encoder=ENCODER_MODEL,
                token_lang=src_lang,
                spm_model="/homes/julez/ts-adapters/src/data/cochrane/LASER/laser2.spm",
                output=src_emb,
                verbose=False,
            )

            # Farsi needs a different spm model and language code
            if tgt_lang == "fa":
                spm_model = "/homes/julez/ts-adapters/src/data/cochrane/LASER/laser3-pes_Arab.v1.spm"
                tgt_lang = "pes_Arab"
            else:
                spm_model = "/homes/julez/ts-adapters/src/data/cochrane/LASER/laser2.spm"

            embed.embed_sentences(
                ifname=tgt_file,
                encoder=ENCODER_MODEL,
                token_lang=tgt_lang,
                spm_model=spm_model,
                spm_lang=tgt_lang,
                output=tgt_emb,
                verbose=False,
            )

            src_sent2line, src_line_embeddings = dp_utils.read_in_embeddings(src_file, src_emb)
            tgt_sent2line, tgt_line_embeddings = dp_utils.read_in_embeddings(tgt_file, tgt_emb)

            src_max_alignment_size = 1 if one_to_many is not None else alignment_max_size
            tgt_max_alignment_size = one_to_many if one_to_many is not None else alignment_max_size

            width_over2 = ceil(max(src_max_alignment_size, tgt_max_alignment_size) / 2.0) + search_buffer_size

            with open(src_file, "r", encoding="utf-8") as f:
                src_lines = [sent.strip() for sent in f.readlines()]
            with open(tgt_file, "r", encoding="utf-8") as f:
                tgt_lines = [sent.strip() for sent in f.readlines()]

            vecs0 = dp_utils.make_doc_embedding(src_sent2line, src_line_embeddings, src_lines, src_max_alignment_size)
            vecs1 = dp_utils.make_doc_embedding(tgt_sent2line, tgt_line_embeddings, tgt_lines, tgt_max_alignment_size)

            if one_to_many is not None:
                final_alignment_types = dp_utils.make_one_to_many_alignment_types(one_to_many)
            else:
                final_alignment_types = dp_utils.make_alignment_types(alignment_max_size)

            stack = vecalign.vecalign(
                vecs0=vecs0,
                vecs1=vecs1,
                final_alignment_types=final_alignment_types,
                del_percentile_frac=del_percentile_frac,
                width_over2=width_over2,
                max_size_full_dp=max_size_full_dp,
                costs_sample_size=costs_sample_size,
                num_samps_for_norm=num_samps_for_norm
            )

            return stack[0]["final_alignments"], stack[0]["alignment_scores"]

    #sample = df_scraped[~df_scraped["fr_abstract"].isna()]
    samples = df_scraped[~df_scraped[f"{align_lang}_abstract"].isna()]

    # Iterate over all reviews and find pairs of corresponding sentences in en-target language
    pairs = []
    for doi, row in tqdm(samples.iterrows(), total=len(samples)):
        src_sents = list(dict.fromkeys(row["en_abstract"]))
        tgt_sents = list(dict.fromkeys(row[f"{align_lang}_abstract"]))
        alignments, scores = align(src_sents, tgt_sents, src_lang="en", tgt_lang=align_lang)

        for src_ids, tgt_ids in alignments:
            if len(src_ids) == 1 and len(tgt_ids) == 1:
                pairs.append((
                    doi,
                    src_sents[src_ids[0]],
                    tgt_sents[tgt_ids[0]]
                ))

    df_complex_en_tgt = pd.DataFrame(pairs, columns=["doi", "complex_en", f"complex_{align_lang}"])

    filtered = df_alignments[df_alignments["doi"].isin(set(df_complex_en_tgt["doi"])) & (df_alignments["lang"] == align_lang)]

    merged = pd.merge(
        filtered,
        df_complex_en_tgt,
        left_on=["doi", "input_text"],
        right_on=["doi", "complex_en"],
        how="left"
    )

    merged = merged.drop(["prefix", "input_text", "complex_en"], axis=1)
    merged = merged[["doi", "lang", "split", f"complex_{align_lang}", "target_text"]]
    merged.rename(columns={f"complex_{align_lang}": "src", "target_text": "tgt"}, inplace=True)

    df_merged_train = merged[merged["split"] == "train"]
    df_merged_test = merged[merged["split"] == "test"]
    df_merged_validation = merged[merged["split"] == "val"]

    file_path = f"/homes/julez/ts-adapters/src/data/cochrane_multi_sent/{align_lang}"
    if not path.exists(file_path):
        makedirs(file_path)

    df_merged_train.to_csv(f"{file_path}/train_aligned.csv", index=False)
    df_merged_test.to_csv(f"{file_path}/test_aligned.csv", index=False)
    df_merged_validation.to_csv(f"{file_path}/validation_aligned.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Crosslingual Alignment", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("lang", type=str, help="Target language to align.")

    args = parser.parse_args()

    main(args.lang)
