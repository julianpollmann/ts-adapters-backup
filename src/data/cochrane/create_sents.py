import argparse

import spacy

from os import path, makedirs
from spacy.language import Language
from tqdm import tqdm
from typing import Protocol
from utils import load_data, convert_doi


def create_data_dirs(output_dir: str, data: list) -> None:
    for article in data:
        if not path.exists(path.join(output_dir, '%s' % convert_doi(article.get('doi')), 'sents')):
            makedirs(path.join(output_dir, '%s' % convert_doi(article.get("doi")), 'sents'))


class NlpFactory(Protocol):
    def get_spacy_object(self) -> Language:
        """Returns a new spacy object belonging to this factory."""


class EnglishNlp:
    def get_spacy_object(self) -> Language:
        return spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer", "textcat"])


class GermanNlp:
    def get_spacy_object(self) -> Language:
        return spacy.load("de_core_news_sm", exclude=["ner", "lemmatizer", "textcat"])


class SpanishNlp:
    def get_spacy_object(self) -> Language:
        return spacy.load("es_core_news_sm", exclude=["ner", "lemmatizer", "textcat"])


class FrenchNlp:
    def get_spacy_object(self) -> Language:
        return spacy.load("fr_core_news_sm", exclude=["ner", "lemmatizer", "textcat"])


class CroatianNlp:
    def get_spacy_object(self) -> Language:
        return spacy.load("hr_core_news_sm", exclude=["ner", "lemmatizer", "textcat"])


class KoreanNlp:
    def get_spacy_object(self) -> Language:
        return spacy.load("ko_core_news_sm", exclude=["ner", "lemmatizer", "textcat"])


class JapaneseNlp:
    def get_spacy_object(self) -> Language:
        return spacy.load("ja_core_news_sm", exclude=["ner", "lemmatizer", "textcat"])


class FarsiNlp:
    def get_spacy_object(self) -> Language:
        nlp = spacy.blank('fa')
        nlp.add_pipe('sentencizer')

        return nlp


class PortugueseNlp:
    def get_spacy_object(self) -> Language:
        return spacy.load("pt_core_news_sm", exclude=["ner", "lemmatizer", "textcat"])


class RussianNlp:
    def get_spacy_object(self) -> Language:
        return spacy.load("ru_core_news_sm", exclude=["ner", "lemmatizer", "textcat"])


class ThaiNlp:
    def get_spacy_object(self) -> Language:
        nlp = spacy.blank('th')
        nlp.add_pipe('pythainlp')

        return nlp


class ChineseNlp:
    def get_spacy_object(self) -> Language:
        return spacy.load("zh_core_web_sm", exclude=["ner", "lemmatizer", "textcat"])


def create_nlp_factories():
    chinese_nlp = ChineseNlp()
    factories = {
        "de": GermanNlp(),
        "en": EnglishNlp(),
        "es": SpanishNlp(),
        "fa": FarsiNlp(),
        "fr": FrenchNlp(),
        "hr": CroatianNlp(),
        "ja": JapaneseNlp(),
        "ko": KoreanNlp(),
        "ms": None,
        "pt": PortugueseNlp(),
        "ru": RussianNlp(),
        "th": ThaiNlp(),
        "zh_hans": chinese_nlp,
        "zh_hant": chinese_nlp,
    }

    return factories


def get_nlp_factory(factories: dict, lang: str) -> NlpFactory:
    try:
        return factories[lang]
    except KeyError:
        print(f"No model for language {lang} found.")


def create_sent_files(data: list, output_dir: str) -> None:
    #logger.info('Creating sentence files')
    nlps = create_nlp_factories()

    for article in tqdm(data):
        for lang, content in article.get('content').items():
            nlp = get_nlp_factory(nlps, lang)

            # TODO check if file already exists
            abstract = content.get('abstract')
            abstract_sents = get_sents(paragraphs=abstract, lang=lang, nlp=nlp)
            write_sents(abstract_sents, doi=article.get('doi'), output_dir=output_dir, part='abstract', lang=lang, as_json=True)

            pls = content.get('pls')
            pls_sents = get_sents(paragraphs=pls, lang=lang, nlp=nlp)
            write_sents(pls_sents, doi=article.get('doi'), output_dir=output_dir, part='pls', lang=lang, as_json=True)


def main(input_file: str, output_dir: str):
    data = load_data(input_file=input_file)

    create_data_dirs(output_dir=output_dir, data=data)
    create_sent_files(data=data, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Splits text into sentences')
    parser.add_argument(
        '--input_file',
        type=str,
        default='./processed_data/processed_data.json',
        help='Input file containing list of Review dicts.'
    )

    args = parser.parse_args()

    main(input_file=args.input_file, output_dir=args.output_dir)
