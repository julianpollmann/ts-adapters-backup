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
    #nlp['zh'] = nlp_zh
    nlp['zh_hans'] = nlp_zh
    nlp['zh_hant'] = nlp_zh

    return nlp

def replace_special_tokens(text: str, lang: str) -> str:
    # Fix newlines
    text = text.replace('\n', '[NWLNE]')

    # Fix Enumerations for correct sentence tokenization
    text = re.sub(r'((\s|\()\d)(\.)', r'\g<1>[ENUM]', text)

    # Fix Number Seperators for correct sentence tokenization
    if lang in ['zh_hans', 'zh_hant']:
        text = re.sub(r'(\d+)(\.)?(\d+)?', r'[数字]', text) # 数字 = [NUMBER]
    else:
        text = re.sub(r'(\d+)(\.)(\d+)', r'\g<1>[NMB]\g<3>', text)

    return text

def get_doi_dirs(output_dir: str) -> list:
    return [path.join(output_dir, doi_dir) for doi_dir in listdir(output_dir) if path.isdir(path.join(output_dir, doi_dir))]

def get_sents(paragraphs: list, lang: str, nlp: str) -> list:
    sents = []

    # if nlp:
    #     print("Using spacy")
    #     # If spacy language language model available
    #     # if len(paragraphs) == 1:
    #     #     print("LONG paragraph")
    #     #     # Long paragraphs
    #     #     if isinstance(paragraphs[0], dict):
    #     #         text = replace_special_tokens(paragraphs[0]['text'], lang=lang)
    #     #     else:
    #     #         text = replace_special_tokens(paragraphs[0], lang=lang)
            
    #     #     doc = nlp(text)
    #     #     sents = list(doc.sents)
    #     # else:
    #         # Multiple paragraphs
    #         #print("Multi paragraph")

    #     spacy_para = []
    #     for para in paragraphs:
    #         spacy_para.append(replace_special_tokens(para['text'], lang=lang))

    #     for para_idx, doc in enumerate(nlp.pipe(spacy_para)):
    #         for sent_idx, sent in enumerate(doc.sents):
    #             #sents.append({"para_idx": para_idx, "sent_idx": sent_idx, "sent": sent.text})
    #             sents.append(sent.text)
    # else:
        # TODO can be removed
        # if len(paragraphs) == 1:
        #     print("Using NLTK")
        #     # Long paragraphs
        #     if isinstance(paragraphs[0], dict):
        #         text = replace_special_tokens(paragraphs[0]['text'], lang=lang)
        #     else:
        #         text = replace_special_tokens(paragraphs[0], lang=lang)
            
        #     sents = sent_tokenize(text, language=map_nltk_lang(lang))
        # else:

    sent_pos = 0
    for para in paragraphs:
        text = replace_special_tokens(para['text'], lang=lang)

        for sent in sent_tokenize(text, language=map_nltk_lang(lang)):
            sents.append({"para_pos": para['para_pos'], "sent_pos": sent_pos, "text": sent})
            sent_pos = sent_pos + 1
    
    return sents

def write_sents(sents: list, doi: str, output_dir: str, part: str, lang: str, as_json: bool) -> None:
    fname = path.join(output_dir, '%s' % doi.replace('/', '-'), 'sents', f'{part}.{lang}')
    with open(fname, 'a', encoding='utf8') as f:
        for sent in sents:
            f.write(str(sent["text"]) + '\n')

    if as_json:
        with open(f"{fname}.json", 'w', encoding='utf8') as f:
            json.dump(sents, f, ensure_ascii=False, indent=2)

def create_sent_files(data: list, output_dir: str) -> None:
    logger.info('Creating sentence files')
    nlps = create_spacy_objects()

    for article in tqdm(data):
        # if lang:
        #     nlp = nlps.get(lang)

        #     if article.get('content').get(lang):
        #         abstract = article.get('content').get(lang).get('abstract')
        #         abstract_sents = get_sents(paragraphs=abstract, lang=lang, nlp=nlp)
        #         write_sents(abstract_sents, doi=article.get('doi'), output_dir=output_dir, part='abstract', lang=lang, as_json=True)

        #         pls = article.get('content').get(lang).get('pls')
        #         pls_sents = get_sents(paragraphs=pls, lang=lang, nlp=nlp)
        #         write_sents(pls_sents, doi=article.get('doi'), output_dir=output_dir, part='pls', lang=lang, as_json=True)
        # else:
        for lang, content in article.get('content').items():
            nlp = nlps.get(lang)

            # TODO check if file already exists
            abstract = content.get('abstract')
            abstract_sents = get_sents(paragraphs=abstract, lang=lang, nlp=nlp)
            write_sents(abstract_sents, doi=article.get('doi'), output_dir=output_dir, part='abstract', lang=lang, as_json=True)

            pls = content.get('pls')
            pls_sents = get_sents(paragraphs=pls, lang=lang, nlp=nlp)
            write_sents(pls_sents, doi=article.get('doi'), output_dir=output_dir, part='pls', lang=lang, as_json=True)

def create_overlaps(overlaps: int, output_dir: str) -> None:
    logger.info('Creatings overlaps')

    for doi_dir in tqdm(get_doi_dirs(output_dir=output_dir)):
        if not path.exists(path.join(doi_dir, 'overlaps')):
            makedirs(path.join(doi_dir, 'overlaps'))

            for file in listdir(path.join(doi_dir, 'sents')):
                if not file.endswith(".json"):
                    lang = file.split('.')[1]
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
    logger.info('Creating Embeddings')

    encoder_model = embed.SentenceEncoder(model_path=Path(path.join(laser_dir, 'laser2.pt')))

    for doi_dir in tqdm(get_doi_dirs(output_dir=output_dir)):
        if not path.exists(path.join(doi_dir, 'embeddings')):
            makedirs(path.join(doi_dir, 'embeddings'))

        if language:
            abstract_files = [f for f in listdir(path.join(doi_dir, 'overlaps')) if f.startswith('abstract') and not f.endswith("json") and f.split('.')[1] == language]
            pls_files = [f for f in listdir(path.join(doi_dir, 'overlaps')) if f.startswith('pls') and not f.endswith("json") and f.split('.')[1] == language]
        else:
            abstract_files = [f for f in listdir(path.join(doi_dir, 'overlaps')) if f.startswith('abstract') and not f.endswith("json")]
            pls_files = [f for f in listdir(path.join(doi_dir, 'overlaps')) if f.startswith('pls') and not f.endswith("json")]

        create_embeddings(abstract_files, encoder_model=encoder_model, doi_dir=doi_dir, part='abstract')
        create_embeddings(pls_files, encoder_model=encoder_model, doi_dir=doi_dir, part='pls')

def create_alignments(output_dir: str) -> None:
    #./vecalign.py --alignment_max_size 8 --src bleualign_data/dev.de --tgt bleualign_data/dev.fr \
    #--src_embed bleualign_data/overlaps.de bleualign_data/overlaps.de.emb  \
    #--tgt_embed bleualign_data/overlaps.fr bleualign_data/overlaps.fr.emb

    args.alignment_max_size = 4
    args.one_to_many = None
    args.search_buffer_size = 5
    args.del_percentile_frac = 0.2
    args.max_size_full_dp = 300
    args.costs_sample_size = 20000
    args.num_samps_for_norm = 100
    args.print_aligned_text = False
    processing_type = 'abstract'

    doi_dirs = get_doi_dirs(output_dir=output_dir)
    doi_dirs = doi_dirs[:1]

    print(len(doi_dirs))


    for doi_dir in doi_dirs:
        print('#######################################')
        logger.info('Aligning DOI DIR "%s"', doi_dir)

        # Read all overlaps files for processing type
        #embeddings = [f for f in listdir(path.join(doi_dir, 'embeddings')) if f.startswith(processing_type)]
        overlaps = [f for f in listdir(path.join(doi_dir, 'overlaps')) if f.startswith(processing_type)]

        for file in overlaps:
            print("*"*50)
            lang = file.split('.')[1]

            if lang != 'en':
                logger.info('Aligning en to %s', lang)

                # TODO CHECK if files exists in overlaps, sents, embeddings
                # TODO Dont compute alignments if already exists
                try:
                    src_sent2line, src_line_embeddings = dp_utils.read_in_embeddings(f'{doi_dir}/overlaps/{processing_type}_overlaps.en', f'{doi_dir}/embeddings/{processing_type}_overlaps.en.emb')
                    tgt_sent2line, tgt_line_embeddings = dp_utils.read_in_embeddings(f'{doi_dir}/overlaps/{file}', f'{doi_dir}/embeddings/{processing_type}_overlaps.{lang}.emb')

                    src_max_alignment_size = 1 if args.one_to_many is not None else args.alignment_max_size
                    tgt_max_alignment_size = args.one_to_many if args.one_to_many is not None else args.alignment_max_size

                    width_over2 = ceil(max(src_max_alignment_size, tgt_max_alignment_size) / 2.0) + args.search_buffer_size

                    stack_list = []
                    src_file = f'{doi_dir}/sents/{processing_type}.en'
                    tgt_file = f'{doi_dir}/sents/{processing_type}.{lang}'


                    src_lines = open(src_file, 'rt', encoding="utf-8").readlines()
                    vecs0 = dp_utils.make_doc_embedding(src_sent2line, src_line_embeddings, src_lines, src_max_alignment_size)

                    tgt_lines = open(tgt_file, 'rt', encoding="utf-8").readlines()
                    vecs1 = dp_utils.make_doc_embedding(tgt_sent2line, tgt_line_embeddings, tgt_lines, tgt_max_alignment_size)

                    if args.one_to_many is not None:
                        final_alignment_types = dp_utils.make_one_to_many_alignment_types(args.one_to_many)
                    else:
                        final_alignment_types = dp_utils.make_alignment_types(args.alignment_max_size)

                    stack = vecalign.vecalign(vecs0=vecs0,
                                    vecs1=vecs1,
                                    final_alignment_types=final_alignment_types,
                                    del_percentile_frac=args.del_percentile_frac,
                                    width_over2=width_over2,
                                    max_size_full_dp=args.max_size_full_dp,
                                    costs_sample_size=args.costs_sample_size,
                                    num_samps_for_norm=args.num_samps_for_norm)
                    

                    # write final alignments to stdout
                    #dp_utils.print_alignments(stack[0]['final_alignments'], scores=stack[0]['alignment_scores'],
                    #        src_lines=src_lines if args.print_aligned_text else None,
                    #        tgt_lines=tgt_lines if args.print_aligned_text else None)
                    
                    print("#" * 100)
                    for alignment in stack[0]['final_alignments']:
                        print(alignment)

                    # tgt_index = [0]
                    # for alignment in stack[0]['final_alignments']:
                    #     src_alignment = alignment[0]
                    #     if start_index in src_alignment:
                    #         tgt_index = alignment[1]

                    # tgt_sents = tgt_lines[tgt_index[0]:]
                    #for idx, sent in enumerate(tgt_sents):
                    #    print(f"{idx}-{sent}")

                    #[tgt_lines[i].replace('\n', ' ').strip() for i in y]
                    #write_aligned_sents(doi_dir=doi_dir, sents=tgt_sents, part=processing_type, lang=lang)
                
                except Exception as e:
                    print(e)
                    print(doi_dir)

def write_aligned_sents(doi_dir: str, sents: list, part: str, lang: str):
    if not path.exists(path.join(doi_dir, 'aligned_sents')):
        makedirs(path.join(doi_dir, 'aligned_sents'))

    with open(path.join(doi_dir, 'aligned_sents', f'{part}.{lang}'), 'a', encoding='utf8') as f:
        for sent in sents:
            f.write(sent)

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
    parser.add_argument('--input_file', type=str, default='./processed_data/processed_data.json', help='Input file containing list of Review dicts.')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory.')
    parser.add_argument('--language', type=str, default=None, help='Which language should be aligned to english? Defaults to all languages.')

    args = parser.parse_args()

    main(process=args.process, input_file=args.input_file, language=args.language, output_dir=args.output_dir)