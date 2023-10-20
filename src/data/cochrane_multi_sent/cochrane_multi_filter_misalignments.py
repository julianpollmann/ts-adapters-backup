import argparse
import pandas as pd
import spacy

from tqdm import tqdm
from os import path
from hazm import Lemmatizer, word_tokenize

TRESHHOLD = "0.5"
tqdm.pandas()


def get_spacy_model(lang: str):
    models = {
        "en": "en_core_web_trf",
        "es": "es_dep_news_trf",
        "fr": "fr_dep_news_trf"
    }

    return spacy.load(models.get(lang))


def main(lang: str):
    df = pd.read_csv(path.join(lang, "train.csv"))

    if lang in ["en", "es", "fr"]:
        nlp = get_spacy_model(lang)
        nlp.get_pipe("lemmatizer")
    
        def compute_overlap(src, tgt):
            src_doc = nlp(src)
            src_lemmas = [token.lemma_.lower() for token in src_doc]
    
            tgt_doc = nlp(tgt)
            tgt_lemmas = [token.lemma_.lower() for token in tgt_doc]
    
            intersection = [x for x in src_lemmas if x in tgt_lemmas]
            overlap = len(intersection) / len(tgt_lemmas)
    
            return overlap
        
    if lang == "fa":
        farsi_lemmatizer = Lemmatizer()

        def compute_overlap(src, tgt):
            src_lemmas = [farsi_lemmatizer.lemmatize(token) for token in word_tokenize(src)]
            tgt_lemmas = [farsi_lemmatizer.lemmatize(token) for token in word_tokenize(tgt)]

            intersection = [x for x in src_lemmas if x in tgt_lemmas]
            overlap = len(intersection) / len(tgt_lemmas)

            return overlap

    df["overlap"] = df.progress_apply(lambda row: compute_overlap(row["src"], row["tgt"]), axis=1)
    df["overlap"] = df["overlap"].astype(float)
    df_filtered = df.loc[df['overlap'] >= 0.5]
    df_filtered.to_csv(path.join(lang, "train_filtered.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Filter Misalignments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("lang", type=str, help="Target language to align.")

    args = parser.parse_args()

    main(args.lang)
