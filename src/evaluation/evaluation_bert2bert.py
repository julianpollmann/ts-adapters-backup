import argparse
import os.path

import numpy as np
import pandas as pd
from datasets import load_dataset
from easse.report import write_html_report, DEFAULT_METRICS
from evaluate import load
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, EncoderDecoderModel, AdapterConfig
from transformers.adapters import Stack


def save_data(path: str, fn: str, data: list):
    with open(os.path.join(path, fn), "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in data)


def get_text_len(dataset, text, tokenizer):
    tokenized_targets_train = dataset["test"].map(
        lambda x: tokenizer(x[text], truncation=True),
        batched=True,
        remove_columns=["src", "tgt"],
    )
    tokenized_targets_valid = dataset["test"].map(
        lambda x: tokenizer(x[text], truncation=True),
        batched=True,
        remove_columns=["src", "tgt"],
    )
    train_target_length = [len(x) for x in tokenized_targets_train["input_ids"]]
    valid_target_length = [len(x) for x in tokenized_targets_valid["input_ids"]]

    return max(train_target_length + valid_target_length)


def main(data_args: argparse.Namespace):
    dataset = load_dataset(data_args.dataset, name=data_args.lang_dataset)

    tokenizer = AutoTokenizer.from_pretrained(data_args.base_model)
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    if data_args.adapter_path:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            data_args.base_model,
            data_args.base_model,
            tie_encoder_decoder=True
        )

        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size

        bad_words = ['[CLS]']
        bad_words_ids = [tokenizer.vocab[token] for token in bad_words]
        model.config.bad_words_ids = [bad_words_ids]

        if data_args.lang_adapter:
            # Language adapter
            lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
            model.load_adapter(
                f"{data_args.lang_adapter}/wiki@ukp",
                config=lang_adapter_config,
                model_name=data_args.base_model
            )

        # Task adapter
        adapter_name = model.load_adapter(data_args.adapter_path, config="pfeiffer")

        # Adapter composition + activation
        if data_args.lang_adapter:
            model.active_adapters = Stack(data_args.lang_adapter, adapter_name)
        else:
            model.set_active_adapters(adapter_name)
    else:
        model = EncoderDecoderModel.from_pretrained(data_args.checkpoint)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    max_source_length = get_text_len(dataset, "src", tokenizer)
    max_target_length = get_text_len(dataset, "tgt", tokenizer)

    def preprocess_function(examples, padding="max_length"):
        model_inputs = tokenizer(
            examples["src"],
            max_length=max_source_length,
            padding=padding,
            truncation=True
        )
        labels = tokenizer(
            text_target=examples["tgt"],
            max_length=max_target_length,
            padding=padding,
            truncation=True
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    column_names = dataset["test"].column_names
    test_ds = dataset["test"].map(preprocess_function, remove_columns=column_names, batched=True)
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    #bleu = load("bleu")
    ter = load("ter")

    def compute_metrics(eval_pred):
        predictions, labels, inputs = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

        #bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        ter_score = ter.compute(predictions=decoded_preds, references=decoded_labels, case_sensitive=True)
        #sari_score = corpus_sari(orig_sents=decoded_inputs, sys_sents=decoded_preds, refs_sents=[decoded_labels])
        #fkgl_score = corpus_fkgl(sentences=decoded_preds)

        # Write Src/Tgt/Preds to file
        df = pd.DataFrame()
        df["src"] = decoded_inputs
        df["tgt"] = decoded_labels
        df["prediction"] = decoded_preds
        df.to_json(os.path.join(data_args.output_dir, "predictions.json"))

        # Write EASSE HTML Report
        # BLEU, SARI, FKGL are covered by this
        write_html_report(
            f"{data_args.output_dir}/easse_report.html",
            orig_sents=decoded_inputs,
            sys_sents=decoded_preds,
            refs_sents=[decoded_labels],
            test_set="custom",
            lowercase=True,
            tokenizer="13a",
            metrics=DEFAULT_METRICS,
        )

        return {
            #"Bleu": bleu_score.get("bleu"),
            "TER": ter_score.get("score"),
            #"SARI": sari_score,
            #"FKGL": fkgl_score
        }

    trainer_args = Seq2SeqTrainingArguments(
        output_dir=data_args.output_dir,
        per_device_eval_batch_size=8,
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

    metrics = trainer.evaluate(max_length=max_target_length)
    metrics["model_params"] = model.num_parameters()

    trainer.log_metrics(split="test", metrics=metrics)
    trainer.save_metrics(split="test", metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="TS Evaluation", description="TS Evaluation script")
    parser.add_argument("checkpoint", type=str,
                        help="Model Checkpoint. For finetuned models use the path to the trained model, for adapters "
                             "use the path to a base EncoderDecoder model (e.g. ../../models/bert2bert-base-multilingual-cased).")
    parser.add_argument("dataset", type=str, default="../data/cochrane_multi_sent",
                        help="Name of the dataset in Huggingface datasets format. Must contain a test set.")
    parser.add_argument("output_dir", type=str, help="Output dir to store the evaluation.")
    parser.add_argument("--base_model", type=str, default="bert-base-multilingual-cased")
    #parser.add_argument("--max_input_length", type=int, default=512,
    #                    help="Max input length of tokenized text. Defaults to 512")
    #parser.add_argument("--max_target_length", type=int, default=100,
    #                    help="Max input length of tokenized text. Defaults to 512")
    parser.add_argument("--adapter_path", type=str, help="Path to the trained task adapter. When specifying this, "
                                                         "use the base model as checkpoint")
    parser.add_argument("--lang_dataset", type=str, default=None, help="Language of the dataset.")
    parser.add_argument("--lang_adapter", type=str, default=None, help="Language of the language adapter.")

    main(data_args=parser.parse_args())
