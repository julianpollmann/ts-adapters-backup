# ts-adapters
Adapters for text simplification (MSc Julian Pollmann)

## Install conda environments
Install conda environments by their yaml files, located in conda-environments.
E.g.: `conda env create -f finetuning-environment.yml`

There are 5 different environments: `cochrane-scraper`, `data-description`, `finetuning`, `adapters` and `peft` (for LoRA)

## Datasets
Datasets are located in `src/data`

The cochrane folder contains the scraper for the Cochrane library. It contains a seperate README how to use the scraper.
The Newsela folder contains a scripts to process the Newsela dataset with a seperate README.

After dataset creation, these can be loaded with the Huggingface datasets library:\
`ds = load_dataset(data_args.dataset, name=[SOME_LANGUAGE])`\
Leaving out the name parameter will load the English dataset.

Available datasets are:
* Newsela - `load_dataset("newsela")`
* MultiCochrane - `load_dataset("cochrane_multi_sent")`


## Training
All training scripts are located in `src/models`

There are training scripts for:
* Fine-tuning - `finetuning_seq2seq.py`
* Fine-tuning (few-shot) - `finetuning_seq2seq_fewshot.py`
* Fine-tuning (BERT2BERT) - `finetuning_bert2bert.py`
* Adapters - `adapters_seq2seq.py`
* Adapters (BERT2BERT) - `adapters_bert2bert.py`
* LoRA - `lora_seq2seq.py`
			

### Fine-tuning
Run the finetuning training script by:\
`python finetuning_seq2seq.py output_dir dataset`\
See `finetuning-seq2seq.py` for all arguments/options

Few-Shot training:\
`python finetuning_seq2seq_fewshot.py output_dir dataset --languages en es fa --few_shot_language fr`\
You can pass either the pretraining languages or the few-shot language. The portion of the few-shot language can be specified. See `finetuning_seq2seq_fewshot.py` for all arguments

Fine-tuning with bert2bert
`python finetuning_bert2bert.py output_dir dataset`

### Adapters
Standard Adapter training\
`python adapter_seq2seq.py output_dir dataset --reduction_factor 2`\
See `finetuning-seq2seq.py` for all arguments/options. Reduction factor will determine the adapter size.

Adapter training with bert2bert\
`python adapter_bert2bert.py output_dir dataset`

### LoRA
`python lora_seq2seq.py output_dir dataset`


## Evaluation

### Fine-tuning / Adapters
Evaluation script reside in `src/evaluation`\
To run the evaluation for Finetuning and Adapters use the following script:\
`python evaluation_seq2seq.py checkpoint dataset evaluation_output_dir`\
You can specify the checkpoint. In case of finetuning checkpoint is the trained model path, when evaluation Adapters this is
base model (e.g. `facebook/bart-large`) and `--adapter_path` should be specified.

Bert2Bert Evaluation\
`python evaluation_bert2bert.py checkpoint dataset evaluation_output_dir`

### LoRA
`python evaluation_lora_seq2seq.py checkpoint dataset evaluation_output_dir`
