import argparse
from os import path, listdir

import numpy as np
from datasets import load_dataset
from easse.fkgl import corpus_fkgl
from easse.sari import corpus_sari
from evaluate import load
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, logging
from transformers.trainer_utils import get_last_checkpoint

logging.set_verbosity_info()
logger = logging.get_logger("transformers")


def main(data_args: argparse.Namespace):
    dataset = load_dataset(data_args.dataset)

    if data_args.sample:
        dataset = dataset.shuffle(seed=42)
        dataset["train"] = dataset["train"].select(range(data_args.sample))
        dataset["validation"] = dataset["validation"].select(range(round(data_args.sample * 0.1)))

    tokenizer = AutoTokenizer.from_pretrained(data_args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(data_args.checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["src"],
            max_length=data_args.max_input_length,
            truncation=True,
            padding=True
        )
        labels = tokenizer(examples["tgt"], max_length=data_args.max_input_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    column_names = dataset["train"].column_names
    dataset = dataset.map(preprocess_function, remove_columns=column_names, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    bleu = load("bleu")
    ter = load("ter")

    def compute_metrics(eval_pred):
        predictions, labels, inputs = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

        bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        ter_score = ter.compute(predictions=decoded_preds, references=decoded_labels, case_sensitive=True)
        sari_score = corpus_sari(orig_sents=decoded_inputs, sys_sents=decoded_preds, refs_sents=[decoded_labels])
        fkgl_score = corpus_fkgl(sentences=decoded_preds)

        return {
            "Bleu": bleu_score.get("bleu"),
            "TER": ter_score.get("score"),
            "SARI": sari_score,
            "FKGL": fkgl_score
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=data_args.output_dir,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        logging_strategy="steps",
        per_device_train_batch_size=data_args.batch_size,
        per_device_eval_batch_size=data_args.batch_size,
        learning_rate=data_args.learning_rate,
        max_steps=data_args.steps,
        save_total_limit=3,
        optim="adamw_torch",
        predict_with_generate=True,
        include_inputs_for_metrics=True,
        fp16=True,
        eval_steps=100,
        logging_steps=100,
        weight_decay=0.01,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics
    )

    last_checkpoint = None
    if path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()

    metrics = train_result.metrics

    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(dataset["train"])
    )
    metrics["train_samples"] = min(max_train_samples, len(dataset["train"]))

    trainer.log_metrics(split="train", metrics=metrics)
    trainer.save_metrics(split="train", metrics=metrics)
    trainer.save_state()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Text Simplification Finetuning",
                                     description="Finetuning Text Simplification")
    parser.add_argument("output_dir", type=str, help="Output dir to store trained checkpoints")
    parser.add_argument("--checkpoint", type=str, nargs=1, default="facebook/bart-base",
                        help="Model Checkpoint, can be either a pretrained model or a own checkpoint. Defaults to bart-base.")
    parser.add_argument("--steps", type=int, default=400, help="Number of steps to train")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Max input length of tokenized text. Defaults to 512")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--dataset", type=str, default="../datasets/newsela-en",
                        help="Name of the dataset in Huggingface Datasets format")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--learning_rate", type=str, default=2e-5)

    main(data_args=parser.parse_args())
