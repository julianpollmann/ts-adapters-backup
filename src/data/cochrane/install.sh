#!/bin/bash

if ! [ -x "$(command -v git)" ]; then
  echo "Install git first"
  exit 1
fi

if ! [ -x "$(command -v conda)" ]; then
  echo "Install conda first"
  exit 1
fi

git submodule add https://github.com/thompsonb/vecalign.git
git submodule add https://github.com/facebookresearch/LASER.git

conda env create -f ../../../conda-environments/cochrane-scraper.yml

eval "$(conda shell.bash hook)"
conda activate ts-cochrane-scraper

python -m spacy download de_core_news_sm
python -m spacy download hr_core_news_sm
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download ja_core_news_sm
python -m spacy download ko_core_news_sm
python -m spacy download zh_core_web_sm
python -m spacy download ru_core_news_sm
python -m spacy download pt_core_news_sm
python -m spacy download es_core_news_sm

# Thai / Malay
#python -m spacy download xx_ent_wiki_sm

# Farsi?


# TODO
nltk.download('punkt')

# Set env var
export LASER="${PWD}/LASER"

# Download LASER models
( cd LASER ; nllb/download_models.sh pes_Arab )

# Download 3rd party software
( cd LASER ; ./install_external_tools.sh )