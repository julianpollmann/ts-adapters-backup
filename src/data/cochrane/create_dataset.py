import argparse
import json
from os import listdir, makedirs, path
from utils import get_doi_dirs, replace_special_tokens_inverse

def save_json(data: dict, output_dir: str) -> None:
    """Saves data to json files"""
    with open(path.join(output_dir, "final_data.json"), "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def create_src_tgt_pairs(input_dir: str, output_dir: str) -> None:
    """Creates source and target pairs for each language and writes them as json file"""
    data = {"en": [], "es": [], "fr": [], "fa": [], "pt": [], "de": []}
    doi_dirs = get_doi_dirs(input_dir)

    if not path.exists(path.join(output_dir)):
        makedirs(path.join(output_dir))

    for doi_dir in doi_dirs:
        doi = doi_dir.split("/")[-1]

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
                if path.isfile(path.join(doi_dir, "aligned_sents", f"abstract.{lang}")) and path.isfile(path.join(doi_dir, "aligned_sents", f"pls.{lang}")):
                    article = {"doi": doi, "src": "", "tgt": ""}

                    # Complex/Source
                    with open(path.join(doi_dir, "aligned_sents", f"abstract.{lang}"), encoding='utf-8') as f:
                        src = " ".join(f.readlines())
                        article["src"] = replace_special_tokens_inverse(src)

                    # Simple/Target
                    with open(path.join(doi_dir, "aligned_sents", f"abstract.{lang}"), encoding='utf-8') as f:
                        tgt = " ".join(f.readlines())
                        article["tgt"] = replace_special_tokens_inverse(tgt)

                    if article["src"] and article["tgt"]:
                        data[lang].append(article)

    save_json(data, output_dir)

def main(input_dir: str, output_dir: str):
    create_src_tgt_pairs(input_dir, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Sentence alignment using vecalign', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_dir', type=str, default='./data', help='Input file containing list of Review dicts.')
    parser.add_argument('--output_dir', type=str, default='./final_data', help='Output directory.')

    args = parser.parse_args()

    main(input_dir=args.input_dir, output_dir=args.output_dir)