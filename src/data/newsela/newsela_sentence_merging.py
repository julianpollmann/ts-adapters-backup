import pandas as pd
import re
from alignment_utils import get_components
from tqdm import tqdm
from os import path
from nltk.tokenize import RegexpTokenizer

#
# This script is a cli version of the notebook in notebooks/Newsela-Dataset-SentenceMerging.ipynb
#

# Config
# Path to where newsela-auto alignments are
newsela_base_dir = "../../../data/external/newsela-auto/newsela-auto/"
tqdm.pandas()


def main():
    with open(path.join(newsela_base_dir, "all_data/aligned-sentence-pairs-all.tsv")) as file:
        data = []
        for row in file.readlines():
            data.append(row.strip().split('\t'))

    col_names = ['simple_sent_id', 'simple_sent', 'complex_sent_id', 'complex_sent']
    df_aligned = pd.DataFrame(data, columns=col_names)

    def read_split_data(split: str):
        with open(path.join(newsela_base_dir, f"ACL2020/{split}.src")) as file:
            src = []
            for row in file.read().splitlines():
                src.append(row.strip())

        with open(path.join(newsela_base_dir, f"ACL2020/{split}.dst")) as file:
            tgt = []
            for row in file.read().splitlines():
                tgt.append(row.strip())

        return pd.DataFrame({"complex_sent": src, "simple_sent": tgt, "split": split})

    # Read the data and create one DF with complex - simple sents and split
    df_train = read_split_data("train")
    df_test = read_split_data("test")
    df_valid = read_split_data("valid")

    df_splits = pd.concat([df_train, df_test, df_valid], ignore_index=True)

    tokenizer = RegexpTokenizer(r'\w+')
    df_splits["complex_tokenized"] = df_splits.progress_apply(
        lambda row: tokenizer.tokenize(row["complex_sent"]), axis=1)
    df_splits["simple_tokenized"] = df_splits.progress_apply(
        lambda row: tokenizer.tokenize(row["simple_sent"]), axis=1)
    df_splits["hash"] = df_splits.progress_apply(
        lambda row: hash(" ".join(row["complex_tokenized"] + row["simple_tokenized"])), axis=1)

    df_aligned["complex_tokenized"] = df_aligned.progress_apply(
        lambda row: tokenizer.tokenize(row["complex_sent"]), axis=1)
    df_aligned["simple_tokenized"] = df_aligned.progress_apply(
        lambda row: tokenizer.tokenize(row["simple_sent"]), axis=1)
    df_aligned["hash"] = df_aligned.progress_apply(
        lambda row: hash(" ".join(row["complex_tokenized"] + row["simple_tokenized"])), axis=1)

    # Then merge and remove temp columns
    # Rename merged columns
    df_auto_all = df_aligned.merge(df_splits, on="hash")
    df_auto_all = df_auto_all.drop(columns=[
        "complex_tokenized_x",
        "simple_tokenized_x",
        "complex_sent_y",
        "simple_sent_y",
        "complex_tokenized_y",
        "simple_tokenized_y"
    ])
    df_auto_all = df_auto_all.rename(columns={
        "simple_sent_x": "simple_sent",
        "complex_sent_x": "complex_sent"
    })

    # Set article ID
    df_auto_all['article_id'] = df_auto_all.apply(lambda x: re.search('^\w*-?\w*\.?\w*', x['simple_sent_id']).group(),
                                                  axis=1)

    # 666k mentioned in paper
    df_filtered = df_auto_all[df_auto_all['simple_sent'] != df_auto_all['complex_sent']]

    # Set Reading Level Transition
    simple_rl = df_filtered['simple_sent_id'].str.split(pat='-', expand=True)
    complex_rl = df_filtered['complex_sent_id'].str.split(pat='-', expand=True)

    # df_uni['rl_test'] = pd.concat([complex_rl[2], simple_rl[2]], axis = 1).apply(lambda x: '-'.join(x))

    df_filtered['simple_rl'] = simple_rl[2]
    df_filtered['complex_rl'] = complex_rl[2]
    df_filtered['rl_transition'] = df_filtered[['complex_rl', 'simple_rl']].apply(
        lambda row: '-'.join(row.values.astype(str)), axis=1)
    df_filtered = df_filtered.drop(['simple_rl', 'complex_rl'], axis=1)

    # Remove Readability Levels 0-1, 1-2, 2-3
    # Keep 0-2, 0-3, 1-3, 2-3, 0-4, 1-4, 2-4, 3-4
    # 481k mentioned in paper -> 482k
    filters = ['0-1', '1-2', '2-3']
    transitions = ["0-2", "0-3", "0-4", "1-3", "1-4", "2-4", "3-4"]
    df_filtered = df_filtered[~df_filtered.rl_transition.isin(filters)]
    df_filtered["is_aligned"] = True

    dataframes = {}
    for transition in transitions:
        dataframes[f"df_filtered_{transition}"] = df_filtered[df_filtered["rl_transition"] == transition]

    # fmt: <doc_id.lang>-<level>-<par_id>-<sent_id>
    # level 1 == complex, level 0 == simple
    # DOC_ID = lambda x: x.split('-')[-5:-3]
    DOC_ID = lambda x: re.search('^\w*-?\w*\.?\w*', x).group()
    LEVEL = lambda x: x.split('-')[-3]
    SENT_ID = lambda x: int(x.split('-')[-1])  # cast to int for numeric ordering
    IS_COMPLEX = lambda x, y: LEVEL(x) == y
    GET_COMPLEXITY_LVL = lambda x: re.search(r'\d+', x).group()

    def get_corresponding_nodes(components: set, complexity_level) -> list:
        output = []
        for component in components:
            alignment = {"complex": [], "simple": []}

            for key in component:
                if IS_COMPLEX(key, complexity_level):
                    alignment["complex"].append(key)
                else:
                    alignment["simple"].append(key)

            # Assume that Sentence IDs have asc order
            alignment["complex"].sort()
            alignment["simple"].sort()

            output.append(alignment)
        return output

    def merge_sents(nodes: list[dict], df):
        d = {
            "complex_sent_ids": [],
            "complex_sent": [],
            "simple_sent_ids": [],
            "simple_sent": [],
            "split": [],
        }

        for node in tqdm(nodes):
            # Extract + concat complex sents
            d["complex_sent_ids"].append(node["complex"])
            complex_sents = []
            for sent_id in node["complex"]:
                row = df[df["complex_sent_id"] == sent_id]
                complex_sents.append(row["complex_sent"].iloc[0])
            d["complex_sent"].append(" ".join(complex_sents))

            # Extract + concat simple sents
            d["simple_sent_ids"].append(node["simple"])
            simple_sents = []
            for sent_id in node["simple"]:
                row = df[df["simple_sent_id"] == sent_id]
                simple_sents.append(row["simple_sent"].iloc[0])
                split = row["split"].iloc[0]

            d["simple_sent"].append(" ".join(simple_sents))
            d["split"].append(split)

        df_output = pd.DataFrame(data=d)

        return df_output

    # This iterates over DataFrames for all Transitions (Readability Level 0-2, ...)
    # Computes the alignments and gets corresponding sentences
    # All aligned sentences are then added to one DataFrame
    df_out = pd.DataFrame()
    for df_key, orig_df in dataframes.items():
        print("---")
        print(df_key)
        alignments = list(orig_df[['complex_sent_id', 'simple_sent_id', 'is_aligned']].itertuples(index=False))
        components = get_components(alignments)
        nodes = get_corresponding_nodes(components, GET_COMPLEXITY_LVL(df_key))
        df_out = pd.concat([df_out, merge_sents(nodes, orig_df)], ignore_index=True)

    # Write all data
    df_out.to_csv("aligned-sentence-pairs-merged.csv")

    train = df_out[df_out["split"] == "train"]
    test = df_out[df_out["split"] == "test"]
    validation = df_out[df_out["split"] == "valid"]

    train.to_csv("train.csv")
    test.to_csv("test.csv")
    validation.to_csv("valid.csv")


if __name__ == "__main__":
    main()
