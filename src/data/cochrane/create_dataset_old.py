import argparse
import json

from os import path, walk
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
    prog="Cochrane Dataset Creator", description="Creates Cochrane review dataset"
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="./scraped_data/final_data",
    help="Directory of final data",
)
parser.add_argument(
    "--input_file_name",
    type=str,
    default="data_final.json",
    help="Filename of preprocessed data",
)
args = parser.parse_args()

RANDOM_SEED = 42
TEST_SIZE = 0.2
MIN_SAMPLE_SIZE_TEST = 50
MAX_SAMPLE_SIZE_TEST = 500


def read_json(lang: str):
    """Reads json file and return as list"""
    with open(path.join(data_dir, lang, input_file_name), "r") as f:
        return json.load(f)


def concat_parts(parts) -> str:
    """Concatenates parts of abstract or pls"""
    output = ""

    if type(parts) is list:
        for para in parts:
            output += para.get("text") + " "
    else:
        output = parts

    return output.strip()


def replace_newline(text) -> str:
    """Replaces newline with [NWLNE]"""
    return text.replace("\n", "[NWLNE]")


def write_data(articles: list, split: str, language: str) -> None:
    """Writes data to .src, .dst and .doi files"""

    for article in articles:
        (
            doi,
            abstract,
            pls,
        ) = (
            article.get("doi"),
            article.get("abstract"),
            article.get("pls"),
        )

        abstract = replace_newline(concat_parts(abstract))
        pls = replace_newline(concat_parts(pls))

        with open(path.join(data_dir, language, f"{split}.doi"), "a") as f:
            f.write(doi)
            f.write("\n")

        with open(path.join(data_dir, language, f"{split}.src"), "a") as f:
            f.write(abstract)
            f.write("\n")

        with open(path.join(data_dir, language, f"{split}.dst"), "a") as f:
            f.write(pls)
            f.write("\n")


if __name__ == "__main__":
    """Loop over every language and create train/test/dev split

    Drops languages with less than MIN_SAMPLE_SIZE_TEST
    Uses data as testset if sample_size between MIN_SAMPLE_SIZE_TEST and MAX_SAMPLE_SIZE_TEST
    Uses data as train/test/dev split if sample_size larger than MAX_SAMPLE_SIZE_TEST
    Writes split files as .src, .dst and .doi files
    """
    data_dir = args.data_dir
    input_file_name = args.input_file_name

    for subdir, dirs, files in walk(data_dir):
        for dir in dirs:
            lang = dir

            json_data = read_json(lang=lang)
            sample_size = len(json_data)

            # Drop sample if to small
            if sample_size <= MIN_SAMPLE_SIZE_TEST:
                print(f"Skipped {lang} due to small sample size")
                continue

            # Use data as testset if sample_size to small
            # Otherwise create train/test/dev split
            if (
                sample_size > MIN_SAMPLE_SIZE_TEST
                and sample_size <= MAX_SAMPLE_SIZE_TEST
            ):
                write_data(articles=json_data, split="test", language=lang)
                print(f"Wrote {lang} as test set")

            else:
                train, test = train_test_split(
                    json_data, test_size=TEST_SIZE, random_state=RANDOM_SEED
                )
                test, valid = train_test_split(
                    test, test_size=0.5, random_state=RANDOM_SEED
                )

                write_data(articles=train, split="train", language=lang)
                write_data(articles=test, split="test", language=lang)
                write_data(articles=valid, split="valid", language=lang)

                print(f"Wrote {lang} as train/test/dev split")
