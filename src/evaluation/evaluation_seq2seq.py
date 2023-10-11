import argparse
import os.path

import numpy as np
import pandas as pd
from datasets import load_dataset
from easse.report import write_html_report, DEFAULT_METRICS
from evaluate import load
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast
)

MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]
LANGUAGE_MAPPING = {
    "en": "en_XX",
    "es": "es_XX",
    "fa": "fa_IR",
    "fr": "fr_XX"
}

def save_data(path: str, fn: str, data: list):
    with open(os.path.join(path, fn), "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in data)


def main(data_args: argparse.Namespace):
    dataset = load_dataset(data_args.dataset, name=data_args.language)

    tokenizer = AutoTokenizer.from_pretrained(data_args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(data_args.checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        lang = LANGUAGE_MAPPING.get(data_args.language) if data_args.language is not None else LANGUAGE_MAPPING.get("en")

        tokenizer.src_lang = lang
        tokenizer.tgt_lang = lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        # forced_bos_token_id = (
        #     tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        # )
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id[lang]

    if data_args.adapter_path:
        adapter_name = model.load_adapter(data_args.adapter_path, config="pfeiffer")
        model.set_active_adapters(adapter_name)

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["src"],
            max_length=data_args.max_input_length,
            truncation=True,
            padding=True
        )
        labels = tokenizer(
            text_target=examples["tgt"],
            max_length=data_args.max_input_length,
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    column_names = dataset["test"].column_names
    test_ds = dataset["test"].map(preprocess_function, remove_columns=column_names, batched=True)
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    ter = load("ter")

    def compute_metrics(eval_pred):
        predictions, labels, inputs = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

        # bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        ter_score = ter.compute(predictions=decoded_preds, references=decoded_labels, case_sensitive=True)
        # sari_score = corpus_sari(orig_sents=decoded_inputs, sys_sents=decoded_preds, refs_sents=[decoded_labels])
        # fkgl_score = corpus_fkgl(sentences=decoded_preds)

        # Write Src/Tgt/Preds to file
        df = pd.DataFrame()
        df["src"] = decoded_inputs
        df["tgt"] = decoded_labels
        df["prediction"] = decoded_preds
        with open(os.path.join(data_args.output_dir, "predictions.json"), "w", encoding="utf-8") as file:
            df.to_json(file, force_ascii=False)

        df = df[df["prediction"].str.len() > 0]

        # Write EASSE HTML Report
        # BLEU, SARI, FKGL are covered by this
        write_html_report(
            f"{data_args.output_dir}/easse_report.html",
            orig_sents=df["src"].to_list(),
            sys_sents=df["prediction"].to_list(),
            refs_sents=[df["tgt"].to_list()],
            test_set="custom",
            lowercase=True,
            tokenizer="13a",
            metrics=DEFAULT_METRICS,
        )

        # Compute proportion of predictions, which are exaclty like the complex source
        df_equal = df[df["src"] == df["prediction"]]
        non_modified_sentences = round(len(df_equal.index) / len(df), 2)

        return {
            "TER": ter_score.get("score"),
            "NON_MODIFIED": non_modified_sentences,
        }

    trainer_args = Seq2SeqTrainingArguments(
        output_dir=data_args.output_dir,
        auto_find_batch_size=True,
        predict_with_generate=True,
        include_inputs_for_metrics=True,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=trainer_args,
        data_collator=data_collator,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate(max_length=data_args.max_input_length)
    metrics["model_params"] = model.num_parameters()

    trainer.log_metrics(split="test", metrics=metrics)
    trainer.save_metrics(split="test", metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="TS Evaluation", description="TS Evaluation script")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Model Checkpoint. For finetuned models use the path to the trained model, for adapters use the base model"
             "(e.g. facebook/bart-base).")
    parser.add_argument("dataset", type=str, default="../datasets/newsela-en",
                        help="Name of the dataset in Huggingface datasets format. Must contain a test set.")
    parser.add_argument("output_dir", type=str, help="Output dir to store the evaluation.")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Max input length of tokenized text. Defaults to 1024")
    parser.add_argument("--adapter_path", type=str, help="Path to the trained task adapter. When specifying this, "
                                                         "use the base model as checkpoint")
    parser.add_argument("--language", type=str, default=None)

    main(data_args=parser.parse_args())
