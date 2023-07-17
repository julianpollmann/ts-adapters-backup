# Newsela Dataset Loader

Copy the newsela-auto dir obtained from the authors of *Neural CRF Model for Sentence Alignment in Text Simplification*
to `data/external`

## Preprocessing
Run `python newsela_sentence_merging.py` or use the notebook `Newsela-Dataset-SentenceMerging` to generate the sentence
alignments with splits and merges. This will generate train/test/dev splits based on the paper.

Load the Huggingface dataset then by `datasets.load_dataset("newsela")`.

Depending on your current dir, you must may adjust the path e.g., `datasets.load_dataset("src/data/newsela")` from root dir.