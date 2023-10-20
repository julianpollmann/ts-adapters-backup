import re

from os import path, listdir


def get_doi_dirs(output_dir: str) -> list:
    return [path.join(output_dir, doi_dir) for doi_dir in listdir(output_dir) if
            path.isdir(path.join(output_dir, doi_dir))]


def replace_special_tokens(text: str, lang: str) -> str:
    # Fix newlines
    text = text.replace('\n', '[NWLNE]')

    # Fix Enumerations for correct sentence tokenization
    text = re.sub(r'((\s|\()\d)(\.)', r'\g<1>[ENUM]', text)

    # Fix Number Seperators for correct sentence tokenization
    if lang in ['zh_hans', 'zh_hant']:
        text = re.sub(r'(\d+)(\.)?(\d+)?', r'[数字]', text)  # 数字 = [NUMBER]
    else:
        text = re.sub(r'(\d+)(\.)(\d+)', r'\g<1>[NMB]\g<3>', text)

    return text


def replace_special_tokens_inverse(text: str) -> str:
    # Fix newlines
    text = text.replace('[NWLNE]', ' ')
    text = text.replace('\n', ' ')

    # Fix Enumerations for correct sentence tokenization
    text = text.replace('[ENUM]', '.')
    text = text.replace('[NMB]', '.')

    text = text.replace('  ', ' ')

    return text.strip()


def map_nltk_lang(lang: str) -> str:
    languages = {
        'en': 'english',
        'de': 'german',
        'fr': 'french',
        'es': 'spanish',
        'pt': 'portuguese',
    }

    if languages.get(lang):
        return languages.get(lang)

    return 'english'
