import argparse
from typing import List
from math import ceil
from unicodedata import normalize
import tempfile

import pandas as pd
import nltk
import hazm

from tqdm.auto import tqdm
from alignment_utils import get_components, count_alignment_types
from pandas import DataFrame
from os import path, makedirs

LANGUAGES = ["en", "es", "fr", "fa"]
SENTENCE_SPLITTER = {
    "en": lambda x: nltk.sent_tokenize(x, language="english"),
    "es": lambda x: nltk.sent_tokenize(x, language="spanish"),
    "fr": lambda x: nltk.sent_tokenize(x, language="french"),
    "fa": hazm.sent_tokenize
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

        pls = data["pls"]
        pls_flat = []
        if type(pls[0]) == str:
            sents = sent_tokenize(pls[0])
            for sent in sents:
                pls_flat.append(sent)
        else:
            for paragraph in pls:
                sents = sent_tokenize(paragraph["text"])
                for sent in sents:
                    pls_flat.append(sent)

        normalized[f"{lang}_pls"] = pls_flat

    return normalized


def merge_dfs(df_left: DataFrame, df_scraped_src: DataFrame, df_scraped_tgt: DataFrame) -> DataFrame:
    df_merged = pd.merge(
        df_left,
        df_scraped_src,
        how="inner",
        on=["doi", "src"],
    )
    df_merged = pd.merge(
        df_merged,
        df_scraped_tgt,
        how="inner",
        on=["doi", "tgt"],
    )
    df_merged = df_merged.sort_values(by=["doi"])

    return df_merged


def get_corresponding_nodes(components: set, df: DataFrame) -> list:
    src_nodes = df["src_hash"].tolist()
    tgt_nodes = df["tgt_hash"].tolist()

    output = []
    for component in tqdm(components):
        alignment = {"src": [], "tgt": []}

        for key in component:
            if key in src_nodes:
                alignment["src"].append(key)
            elif key in tgt_nodes:
                alignment["tgt"].append(key)

        output.append(alignment)
    return output


def merge_sents(nodes: list, df) -> DataFrame:
    data = {"doi": [], "src": [], "tgt": []}

    for node in tqdm(nodes):
        # Src/Complex
        src_rows = pd.DataFrame()
        for sent_id in node["src"]:
            row = df[df["src_hash"] == sent_id][:1]
            src_rows = pd.concat([src_rows, row], ignore_index=True)
        src_rows = src_rows.sort_values(by=["src_rank"])
        data["src"].append(" ".join(src_rows["src"].tolist()))

        # DOI
        data["doi"].append(src_rows["doi"][0])

        # Tgt/Simple
        tgt_rows = pd.DataFrame()
        for sent_id in node["tgt"]:
            row = df[df["tgt_hash"] == sent_id][:1]
            tgt_rows = pd.concat([tgt_rows, row], ignore_index=True)
        tgt_rows = tgt_rows.sort_values(by="tgt_rank")
        data["tgt"].append(" ".join(tgt_rows["tgt"].tolist()))

    return pd.DataFrame(data)


def main(lang):
    base_path = "/homes/julez/ts-adapters/"

    # Read aligned data
    dataset = {
        "train": pd.read_csv(f"{lang}/train_aligned.csv"),
        "test": pd.read_csv(f"{lang}/test_aligned.csv"),
        "valid": pd.read_csv(f"{lang}/validation_aligned.csv")
    }

    # Read scraped data
    df_scraped = pd.read_json("../cochrane/scraped_data/data.json")
    df_scraped = pd.concat([
        df_scraped,
        pd.json_normalize(df_scraped["content"].apply(normalize_content))
    ], axis=1)
    df_scraped = df_scraped.set_index("doi")

    scraped_data_abstract = {"doi": [], "src": [], "src_rank": []}
    scraped_data_pls = {"doi": [], "tgt": [], "tgt_rank": []}

    for index, row in df_scraped.iterrows():
        abstract = row[f"{lang}_abstract"]
        pls = row[f"{lang}_pls"]

        if type(abstract) is list:
            for sent_rank, sent in enumerate(abstract):
                scraped_data_abstract["doi"].append(index)
                scraped_data_abstract["src"].append(sent)
                scraped_data_abstract["src_rank"].append(sent_rank)

        if type(pls) is list:
            for sent_rank, sent in enumerate(pls):
                scraped_data_pls["doi"].append(index)
                scraped_data_pls["tgt"].append(sent)
                scraped_data_pls["tgt_rank"].append(sent_rank)

    df_scraped_src = pd.DataFrame(scraped_data_abstract)
    df_scraped_tgt = pd.DataFrame(scraped_data_pls)

    # Get sent order foreach split
    for split, df in dataset.items():
        df = merge_dfs(df, df_scraped_src, df_scraped_tgt)

        # Drop Instances where either no src or not tgt
        df = df.dropna()

        # Remove Sentences where src == tgt
        df = df.drop(df[df.src == df.tgt].index)

        # Add hashes as unique ID
        df["src_hash"] = df.apply(lambda x: hash("src" + x["doi"] + str(x["src"])), axis=1)
        df["tgt_hash"] = df.apply(lambda x: hash("tgt" + x["doi"] + str(x["tgt"])), axis=1)

        df["is_aligned"] = True
        dataset[split] = df

    out = {}
    for split, df in dataset.items():
        alignments = list(df[["src_hash", "tgt_hash", "is_aligned"]].itertuples(index=False))
        components = get_components(alignments)
        nodes = get_corresponding_nodes(components, df)

        df_out = merge_sents(nodes, df)

        file_path = f"{base_path}src/data/cochrane_multi_sent/{lang}"
        if not path.exists(file_path):
            makedirs(file_path)
        df_out.to_csv(f"/{file_path}/{split}.csv")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MultiCochrane Sentence Order Merging", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("lang", type=str, help="Target language to align.")

    args = parser.parse_args()

    main(args.lang)
