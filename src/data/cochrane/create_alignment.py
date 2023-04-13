import argparse
import json
import re
import spacy
import logging
import sys
from tqdm import tqdm
from os import path, makedirs, listdir, scandir, environ
from pathlib import Path
from nltk.tokenize import sent_tokenize
from math import ceil

# Import custom modules
sys.path.append(path.abspath('./vecalign/'))
from vecalign import overlap, vecalign, dp_utils

sys.path.append(path.abspath('./LASER/source/'))
laser_dir = path.abspath('./LASER/')
environ["LASER"] = str(laser_dir)
from LASER.source import embed

# Logging
logger = logging.getLogger("ts_adapters")
logger.setLevel(logging.DEBUG)
logFormatter = logging.Formatter("%(asctime)s  %(levelname)-5.5s  %(message)s")
log_filename = './logs/alignment.log'
makedirs(path.dirname(log_filename), exist_ok=True)
fh = logging.FileHandler(log_filename, mode='a')
fh.setLevel(logging.ERROR)
logger.addHandler(fh)

def load_data(input_file: str) -> list:
    with open(input_file, encoding='utf-8') as f:
        return json.load(f)

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

def create_spacy_objects():
    nlp = {}

    nlp_fa = spacy.blank('fa')
    nlp_fa.add_pipe('sentencizer')
    nlp['fa'] = nlp_fa

    nlp_ja = spacy.load("ja_core_news_sm", exclude=["ner", "lemmatizer", "textcat"])
    nlp['ja'] = nlp_ja

    nlp_ko = spacy.load("ko_core_news_sm", exclude=["ner", "lemmatizer", "textcat"])
    nlp['ko'] = nlp_ko

    nlp_th = spacy.blank('th')
    nlp_th.add_pipe('sentencizer')
    nlp['th'] = nlp_th

    nlp_zh = spacy.load('zh_core_web_sm', exclude=["ner", "lemmatizer", "textcat"])
    nlp['zh'] = nlp_zh

    return nlp

def replace_special_tokens(text: str) -> str:
    # Fix newlines
    text = text.replace('\n', '[NWLNE]')

    # Fix Number Seperators for correct sentence tokenization
    # text = re.sub(r'(\d+)(\.)(\d+)', r'\g<1>[NMB SEP]\g<3>', text)

    # Fix Enumerations for correct sentence tokenization
    text = re.sub(r'((\s|\()\d)(\.)', r'\g<1>[ENUM]', text)

    return text

def get_doi_dirs(output_dir: str) -> list:
    return [path.join(output_dir, doi_dir) for doi_dir in listdir(output_dir) if path.isdir(path.join(output_dir, doi_dir))]

def get_sents(paragraphs: list, lang: str, nlp: str) -> list:
    sents = []

    if nlp:
        # If spacy language language model available
        if len(paragraphs) == 1:
            # Long paragraphs
            if isinstance(paragraphs[0], dict):
                text = replace_special_tokens(paragraphs[0]['text'])
            else:
                text = replace_special_tokens(paragraphs[0])
            
            doc = nlp(text)
            sents = list(doc.sents)
        else:
            # Multiple paragraphs
            spacy_para = []
            for para in paragraphs:
                spacy_para.append(replace_special_tokens(para['text']))

            for doc in nlp.pipe(spacy_para):
                for sent in doc.sents:
                    sents.append(sent)
    else:
        if len(paragraphs) == 1:
            # Long paragraphs
            if isinstance(paragraphs[0], dict):
                text = replace_special_tokens(paragraphs[0]['text'])
            else:
                text = replace_special_tokens(paragraphs[0])
            
            sents = sent_tokenize(text, language=map_nltk_lang(lang))
        else:
            # Multiple paragraphs
            for para in paragraphs:
                text = replace_special_tokens(para['text'])

                for sent in sent_tokenize(text, language=map_nltk_lang(lang)):
                    sents.append(sent)
    
    return sents

def write_sents(sents: list, doi: str, output_dir: str, part: str, lang: str) -> None:
    with open(path.join(output_dir, '%s' % doi.replace('/', '-'), 'sents', f'{part}.{lang}'), 'a', encoding='utf8') as f:
        for sent in sents:
            f.write(str(sent) + '\n')

def create_sent_files(data: list, output_dir: str) -> None:
    logger.info('Creating sentence files')
    nlps = create_spacy_objects()

    for article in tqdm(data):
        for lang, content in article.get('content').items():
            nlp = nlps.get(lang)

            # TODO check if file already exists
            abstract = content.get('abstract')
            abstract_sents = get_sents(paragraphs=abstract, lang=lang, nlp=nlp)
            write_sents(abstract_sents, doi=article.get('doi'), output_dir=output_dir, part='abstract', lang=lang)

            pls = content.get('pls')
            pls_sents = get_sents(paragraphs=pls, lang=lang, nlp=nlp)
            write_sents(pls_sents, doi=article.get('doi'), output_dir=output_dir, part='pls', lang=lang)

def create_overlaps(overlaps: int, output_dir: str) -> None:
    logger.info('Creatings overlaps')

    for doi_dir in tqdm(get_doi_dirs(output_dir=output_dir)):
        if not path.exists(path.join(doi_dir, 'overlaps')):
            makedirs(path.join(doi_dir, 'overlaps'))

            for file in scandir(path.join(doi_dir, 'sents')):                    
                lang = file.name.split('.')[1]
                overlap.go(output_file=f'{doi_dir}/overlaps/abstract_overlaps.{lang}',
                        input_files=[f'{doi_dir}/sents/abstract.{lang}'], num_overlaps=overlaps)
                
                overlap.go(output_file=f'{doi_dir}/overlaps/pls_overlaps.{lang}',
                        input_files=[f'{doi_dir}/sents/pls.{lang}'], num_overlaps=overlaps)                  

def create_embeddings(files: list, encoder_model, doi_dir: str, part: str) -> None:
    for file in files:
        ext_lang = file.split('.')[1]
        token_lang = ext_lang

        # Normalize ISO language codes
        if ext_lang in ['zh_hans', 'zh_hant']:
            token_lang = 'zh'

        try:
            embed.embed_sentences(
                ifname=f'{doi_dir}/overlaps/{part}_overlaps.{ext_lang}',
                encoder=encoder_model,
                token_lang=token_lang,
                spm_model=path.join(laser_dir, 'laser2.spm'),
                output=Path(f'{doi_dir}/embeddings/{part}_overlaps.{ext_lang}.emb'),
                verbose=True,
            )
        except Exception as e:
            logger.error(e)

def create_embedding_files(language: str, output_dir: str) -> None:
    logger.info('Creatings Embeddings')

    encoder_model = embed.SentenceEncoder(model_path=Path(path.join(laser_dir, 'laser2.pt')))

    for doi_dir in tqdm(get_doi_dirs(output_dir=output_dir)):
        if not path.exists(path.join(doi_dir, 'embeddings')):
            makedirs(path.join(doi_dir, 'embeddings'))

        if language:
            abstract_files = [f for f in listdir(path.join(doi_dir, 'overlaps')) if f.startswith('abstract') and f.split('.')[1] == language]
            pls_files = [f for f in listdir(path.join(doi_dir, 'overlaps')) if f.startswith('pls') and f.split('.')[1] == language]
        else:
            abstract_files = [f for f in listdir(path.join(doi_dir, 'overlaps')) if f.startswith('abstract')]
            pls_files = [f for f in listdir(path.join(doi_dir, 'overlaps')) if f.startswith('pls')]

        create_embeddings(abstract_files, encoder_model=encoder_model, doi_dir=doi_dir, part='abstract')
        create_embeddings(pls_files, encoder_model=encoder_model, doi_dir=doi_dir, part='pls')

def create_alignments(output_dir: str) -> None:
    #./vecalign.py --alignment_max_size 8 --src bleualign_data/test*.de --tgt bleualign_data/test*.fr \
    #--gold bleualign_data/test*.defr  \
    #--src_embed bleualign_data/overlaps.de bleualign_data/overlaps.de.emb  \ ## CHECK!
    #--tgt_embed bleualign_data/overlaps.fr bleualign_data/overlaps.fr.emb > /dev/null ## CHECK!

    args.alignment_max_size = 8
    args.one_to_many = None
    args.search_buffer_size = 5

    for doi_dir in get_doi_dirs(output_dir=output_dir):
        for file in listdir(path.join(doi_dir, 'overlaps')):
            print('---')
            # TODO GET all files except english
            lang = file.split('.')[1]

            if lang != 'en':
                # TODO CHECK if files in overlaps, sents, embeddings

                # This is taken from vecalign.py
                try:
                    src_sent2line, src_line_embeddings = dp_utils.read_in_embeddings(f'{doi_dir}/overlaps/overlaps.en', f'{doi_dir}/embeddings/overlaps.en.emb')
                    tgt_sent2line, tgt_line_embeddings = dp_utils.read_in_embeddings(f'{doi_dir}/overlaps/{file}', f'{doi_dir}/embeddings/overlaps.{lang}.emb')
                except Exception as e:
                    print(e)
                    print(doi_dir)

                src_max_alignment_size = 1 if args.one_to_many is not None else args.alignment_max_size
                tgt_max_alignment_size = args.one_to_many if args.one_to_many is not None else args.alignment_max_size

                width_over2 = ceil(max(src_max_alignment_size, tgt_max_alignment_size) / 2.0) + args.search_buffer_size

                test_alignments = []
                stack_list = []
                src_file = f'{doi_dir}/sents/dev.en'
                tgt_file = f'{doi_dir}/sents/dev.{lang}'
                for src_file, tgt_file in zip(src_file, tgt_file):
                    logger.info('Aligning src="%s" to tgt="%s"', src_file, tgt_file)

                    # TODO Fix this
                    src_lines = open(src_file, 'rt', encoding="utf-8").readlines()
                    vecs0 = dp_utils.make_doc_embedding(src_sent2line, src_line_embeddings, src_lines, src_max_alignment_size)

                    tgt_lines = open(tgt_file, 'rt', encoding="utf-8").readlines()
                    vecs1 = dp_utils.make_doc_embedding(tgt_sent2line, tgt_line_embeddings, tgt_lines, tgt_max_alignment_size)

                    if args.one_to_many is not None:
                        final_alignment_types = dp_utils.make_one_to_many_alignment_types(args.one_to_many)
                    else:
                        final_alignment_types = dp_utils.make_alignment_types(args.alignment_max_size)
                    logger.debug('Considering alignment types %s', final_alignment_types)

                    stack = vecalign(vecs0=vecs0,
                                    vecs1=vecs1,
                                    final_alignment_types=final_alignment_types,
                                    del_percentile_frac=args.del_percentile_frac,
                                    width_over2=width_over2,
                                    max_size_full_dp=args.max_size_full_dp,
                                    costs_sample_size=args.costs_sample_size,
                                    num_samps_for_norm=args.num_samps_for_norm)

                    # write final alignments to stdout
                    dp_utils.print_alignments(stack[0]['final_alignments'], scores=stack[0]['alignment_scores'],
                                    src_lines=src_lines if args.print_aligned_text else None,
                                    tgt_lines=tgt_lines if args.print_aligned_text else None)

                    test_alignments.append(stack[0]['final_alignments'])
                    stack_list.append(stack)

def create_data_dirs(output_dir: str, data: list) -> None:
    logger.info('Creating data dirs')

    for article in data:
        if not path.exists(path.join(output_dir, '%s' % article.get('doi').replace('/', '-'), 'sents')):
            makedirs(path.join(output_dir, '%s' % article.get('doi').replace('/', '-'), 'sents'))

def main(process: str, input_file: str, language: str, output_dir: str):
    if process == 'sents':
        data = load_data(input_file=input_file)

        create_data_dirs(output_dir=output_dir, data=data)
        create_sent_files(data=data, output_dir=output_dir)

    if process == 'overlaps':
        create_overlaps(overlaps=10, output_dir=output_dir)

    if process == 'embeddings':
        create_embedding_files(language=language, output_dir=output_dir)

    if process == 'alignments':
        create_alignments(output_dir=output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Sentence alignment using vecalign', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("process", choices=["sents", "overlaps", "embeddings", "alignments"], help="Which process should be run)?")
    parser.add_argument('--input_file', type=str, help='Input file containing list of Review dicts.')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory.')
    parser.add_argument('--language', type=str, default=None, help='Which language should be aligned to english? Defaults to all languages.')

    args = parser.parse_args()

    main(process=args.process, input_file=args.input_file, language=args.language, output_dir=args.output_dir)