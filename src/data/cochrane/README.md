## Setup
Prerequisites: installed git and conda.

```sh
chmod +ax install.sh
./install.sh
```

This will install all dependencies via conda, clone [vecalign](https://github.com/thompsonb/vecalign)
and [laser](https://github.com/facebookresearch/LASER) git repositories and
downloads multiple spaCy language models. Activate the conda environment by
```sh
conda activate ts-cochrane-scraper
```

## Construct the dataset

### Scraping
To construct the dataset first scrape all the data from the Cochrane Website.

#### 1. Scrape DOIs of all reviews
```sh
python scrape.py dois
```
DOIs are written to `./scraped_data/dois.txt`. You can specify the output_dir and specific DOIS to scrape.

#### 2. Scrape all reviews by
```sh
python scrape.py articles
```
This will download all metadata for the reviews (e.g. the links for all languages).
Each review is written to a json file.

##### 3. Scrape the content for all languages by 
```sh
python scrape.py content
```
This will download the content (abstract and plain language summary) of each review.
It will take some time, since for each review all languages will be scraped.
Every review is written to their json file; all reviews are written to `./scraped_data/data.json` by default.


### Preprocess the data
Run `python process.py` to preprocess and filter the data. By default following filters are applied:


### Create alignments
This will create the alignments between english truncated abstracts/pls and the abstracts/pls of all other languages.

#### 1. Create Sents and Overlaps for each language
```sh
python create_alignment.py sents
python create_alignment.py overlaps
```

#### 2. Create the embeddings
To create the embeddings specify the CUDA device and set a language to create embeddings in parallel for
multiple languages.
```sh
CUDA_VISIBLE_DEVICES=[YOUR_CUDA_DEV_NO] python create_alignment.py embeddings --language en
```

#### 3. Create alignments
TBD 
```sh
python create_alignment.py alignments
```