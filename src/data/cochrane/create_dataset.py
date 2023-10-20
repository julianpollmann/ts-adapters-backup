import argparse
import json
from os import listdir, makedirs, path
from utils import get_doi_dirs, replace_special_tokens_inverse
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TEST_SIZE = 0.2
VALID_SIZE = 0.5
MIN_SAMPLE_SIZE_TEST = 50
MAX_SAMPLE_SIZE_TEST = 500


def save_json(data: dict, output_dir: str, filename: str) -> None:
    """Saves data to json files"""

    if not path.exists(path.join(output_dir)):
        makedirs(path.join(output_dir))

    with open(path.join(output_dir, filename), "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_data(articles: list, split: str, language: str, output_dir: str) -> None:
    """Writes data to .src, .dst and .doi files for each language.

    Keyword arguments:
    articles -- list of articles
    split -- train, test or valid
    language -- language of articles/dir to save data
    output_dir -- dir to save data
    """

    if not path.exists(path.join(output_dir, language)):
        makedirs(path.join(output_dir, language))

    for article in articles:
        doi, abstract, pls = article.get("doi"), article.get("src"), article.get("tgt")

        with open(path.join(output_dir, language, f"{split}.doi"), "a") as f:
            f.write(doi)
            f.write("\n")

        with open(path.join(output_dir, language, f"{split}.src"), "a") as f:
            f.write(abstract)
            f.write("\n")

        with open(path.join(output_dir, language, f"{split}.dst"), "a") as f:
            f.write(pls)
            f.write("\n")


def save_jsonl(articles: list, split: str, output_dir: str, language: str):
    if not path.exists(path.join(output_dir, language)):
        makedirs(path.join(output_dir, language))

        # doi, abstract, pls = article.get("doi"), article.get("src"), article.get("tgt")

    with open(path.join(output_dir, language, f"{split}.jsonl"), "w") as f:
        for article in articles:
            json.dump(article, f, ensure_ascii=False)
            f.write("\n")


def create_src_tgt_pairs_lang(input_dir: str) -> dict:
    """Creates source and target pairs for each language.

    Reads every dir of aligned data and first creates english language pairs.
    Then creates pairs for other languages specified in data dict.
    Pairs are constructed using the (aligned) source sents and the (aligned)
    pls sents and converted tokens from a previous step are converted back.

    Keyword arguments:
    input_dir -- dir of aligned data
    """

    data = {"en": [], "es": [], "fr": [], "fa": [], "pt": [], "de": []}
    doi_dirs = get_doi_dirs(input_dir)

    for doi_dir in doi_dirs:
        doi = doi_dir.split("/")[-1]

        # TODO remove this
        if (path.isfile(path.join(doi_dir, 'sents/abstract.en')) and
                path.isfile(path.join(doi_dir, 'sents/pls.en'))):

            en_article = {"doi": doi, "src": "", "tgt": ""}

            # English Complex/Source
            with open(path.join(doi_dir, 'sents/abstract.en'), encoding='utf-8') as f:
                src_en = " ".join(f.readlines())
                en_article["src"] = replace_special_tokens_inverse(src_en)

            # English Simple/Target
            with open(path.join(doi_dir, 'sents/pls.en'), encoding='utf-8') as f:
                tgt_en = " ".join(f.readlines())
                en_article["tgt"] = replace_special_tokens_inverse(tgt_en)

            if en_article["src"] and en_article["tgt"]:
                data["en"].append(en_article)

            # Other languages
            for lang in data:
                if lang != "en":
                    try:
                        article = {"doi": doi, "src": "", "tgt": ""}

                        # Complex/Source
                        with open(path.join(doi_dir, "aligned_sents", f"abstract.{lang}"), encoding='utf-8') as f:
                            src = " ".join(f.readlines())
                            article["src"] = replace_special_tokens_inverse(src)

                        # Simple/Target
                        with open(path.join(doi_dir, "aligned_sents", f"pls.{lang}"), encoding='utf-8') as f:
                            tgt = " ".join(f.readlines())
                            article["tgt"] = replace_special_tokens_inverse(tgt)

                        if article["src"] and article["tgt"]:
                            data[lang].append(article)
                    except FileNotFoundError:
                        print(f"For DOI {doi} and Language {lang}: abstract or pls missing")
                        continue
        else:
            print(f"Abstract or PLS not found: {doi}")

    return data


def create_src_tgt_pairs(input_dir: str) -> list:
    doi_dirs = get_doi_dirs(input_dir)

    data = []
    languages = ["en", "es", "fr", "fa", "pt", "de"]

    for doi_dir in doi_dirs:
        doi = doi_dir.split("/")[-1]

        en_article = {"doi": doi, "lang": "en", "src": "", "tgt": ""}

        # English Complex/Source
        with open(path.join(doi_dir, 'sents/abstract.en'), encoding='utf-8') as f:
            src_en = " ".join(f.readlines())
            en_article["src"] = replace_special_tokens_inverse(src_en)

        # English Simple/Target
        with open(path.join(doi_dir, 'sents/pls.en'), encoding='utf-8') as f:
            tgt_en = " ".join(f.readlines())
            en_article["tgt"] = replace_special_tokens_inverse(tgt_en)

        if en_article["src"] and en_article["tgt"]:
            data.append(en_article)

        # Other languages
        for lang in languages:
            if lang != "en":
                try:
                    article = {"doi": doi, "lang": lang, "src": "", "tgt": ""}

                    print(doi_dir)

                    # Complex/Source
                    with open(path.join(doi_dir, "aligned_sents", f"abstract.{lang}"), encoding='utf-8') as f:
                        src = " ".join(f.readlines())
                        article["src"] = replace_special_tokens_inverse(src)

                    # Simple/Target
                    with open(path.join(doi_dir, "aligned_sents", f"pls.{lang}"), encoding='utf-8') as f:
                        tgt = " ".join(f.readlines())
                        article["tgt"] = replace_special_tokens_inverse(tgt)

                    if article["src"] and article["tgt"]:
                        data.append(article)
                except FileNotFoundError:
                    print(f"For DOI {doi} and Language {lang}: abstract or pls missing")
                    continue

    return data


def create_train_test_split(data: dict, output_dir: str) -> None:
    """Creates train/test/valid split for each language and saves the files.

    If sample size is to small, the data is excluded. If sample size is between
    MIN_SAMPLE_SIZE and MAX_SAMPLE_SIZE, the data is used as test set.
    If sample size larger than MAX_SAMPLE_SIZE, the data is split into
    train/test/valid.

    Keyword arguments:
    data -- data to split (dict of lang: [articles])
    output_dir -- dir to write the data
    """

    final_data_split = {}
    for lang, articles in data.items():
        sample_size = len(articles)

        # Drop sample if to small
        if sample_size <= MIN_SAMPLE_SIZE_TEST:
            print(f"Skipped {lang} due to small sample size")
            continue

        # Use data as testset if sample_size to small
        # Otherwise create train/test/valid split
        if MIN_SAMPLE_SIZE_TEST < sample_size <= MAX_SAMPLE_SIZE_TEST:
            save_data(articles=articles, split="test", language=lang, output_dir=output_dir)
            save_jsonl(articles=articles, split="test", language=lang, output_dir=output_dir)
            final_data_split[lang] = {"test": articles}

            print(f"Wrote {lang} as test set")

        else:
            train, test = train_test_split(articles, test_size=TEST_SIZE, random_state=RANDOM_SEED)
            test, valid = train_test_split(test, test_size=VALID_SIZE, random_state=RANDOM_SEED)

            save_data(articles=train, split="train", language=lang, output_dir=output_dir)
            save_data(articles=test, split="test", language=lang, output_dir=output_dir)
            save_data(articles=valid, split="valid", language=lang, output_dir=output_dir)

            save_jsonl(articles=train, split="train", language=lang, output_dir=output_dir)
            save_jsonl(articles=test, split="test", language=lang, output_dir=output_dir)
            save_jsonl(articles=valid, split="valid", language=lang, output_dir=output_dir)

            final_data_split[lang] = {"train": train, "test": test, "valid": valid}

            print(f"Wrote {lang} as train/test/valid split")

    save_json(final_data_split, output_dir, "final_data_split_lang.json")


def main(input_dir: str, output_dir: str):
    # data = create_src_tgt_pairs(input_dir)
    data = create_src_tgt_pairs_lang(input_dir)
    save_json(data, output_dir, "final_data.json")

    create_train_test_split(data, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Create Dataset for Text Simplification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=str, default='./aligned_data',
                        help='Input file containing list of Review dicts.')
    parser.add_argument('--output_dir', type=str, default='./final_data', help='Output directory.')

    args = parser.parse_args()

    main(input_dir=args.input_dir, output_dir=args.output_dir)
